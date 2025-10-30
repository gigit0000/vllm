# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend

import torch

from vllm.attention.backends.abstract import AttentionMetadata
from vllm.config import CacheConfig, ModelConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.utils import direct_register_custom_op
from vllm.v1.attention.backends.short_conv_attn import ShortConvAttentionMetadata


@CustomOp.register("short_conv")
class ShortConv(MambaBase, CustomOp):
    def __init__(
        self,
        config,
        dim: int,
        layer_idx: int,
        model_config: Optional[ModelConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.conv_dim = dim
        self.L_cache = config.conv_L_cache
        self.bias = config.conv_bias

        self.conv = ColumnParallelLinear(
            input_size=self.L_cache,
            output_size=dim,
            bias=self.bias,
            prefix=f"{prefix}.conv1d",
        )
        # unsqueeze to fit conv1d weights shape into the linear weights shape.
        # Can't do this in `weight_loader` since it already exists in
        # `ColumnParallelLinear` and `set_weight_attrs`
        # doesn't allow to override it
        self.conv.weight.data = self.conv.weight.data.unsqueeze(1)

        self.in_proj = MergedColumnParallelLinear(
            input_size=dim,
            output_sizes=[dim] * 3,
            bias=self.bias,
            prefix=f"{prefix}.in_proj",
        )
        self.out_proj = RowParallelLinear(
            input_size=dim,
            output_size=dim,
            bias=self.bias,
            prefix=f"{prefix}.out_proj",
        )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.kv_cache = (torch.tensor([]),)

        self.model_config = model_config
        self.cache_config = cache_config
        self.prefix = prefix

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        return

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        torch.ops.vllm.short_conv(
            hidden_states,
            output,
            self.prefix,
        )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        forward_context = get_forward_context()
        # ShortConvAttentionMetadata contains metadata necessary for the
        # short_conv triton kernels to operate in continuous batching and in
        # chunked prefill modes; they are computed at top-level model forward
        # since they stay the same and reused for all mamba layers in the same
        # iteration.
        attn_metadata: AttentionMetadata = forward_context.attn_metadata
        
        assert self.cache_config is not None
        mamba_block_size = self.cache_config.mamba_block_size
        print("\nshort_conv.py - mamba_block_size", mamba_block_size)   
        prefix_caching_enabled = self.cache_config.enable_prefix_caching             
        if attn_metadata is not None:
            assert isinstance(attn_metadata, dict)
            attn_metadata = attn_metadata[self.prefix]
            assert isinstance(attn_metadata, ShortConvAttentionMetadata)
            self_kv_cache = self.kv_cache[forward_context.virtual_engine]
            conv_state = self_kv_cache[0].transpose(-1, -2)
            state_indices_tensor = attn_metadata.state_indices_tensor
            has_initial_states_p = attn_metadata.has_initial_states_p
            
            #WILL 추가
            # 이건 ssm용
            # prep_initial_states = attn_metadata.prep_initial_states
            # chunk_size = attn_metadata.chunk_size
            #이게 필요한가??
            seq_idx_p = attn_metadata.seq_idx_p
            query_start_loc_p = attn_metadata.query_start_loc_p
            #이건 ssm용
            # cu_chunk_seqlen_p = attn_metadata.cu_chunk_seqlen_p
            # last_chunk_indices_p = attn_metadata.last_chunk_indices_p            

        if 'query_start_loc_p' in locals():
            print("\n이게이상query_start_loc_p", query_start_loc_p)
        BCx, _ = self.in_proj(hidden_states)

        B, C, x = BCx.chunk(3, dim=-1)

        conv_weights = self.conv.weight.view(
            self.conv.weight.size(0), self.conv.weight.size(2)
        )

        if attn_metadata is None:
            # V1 profile run
            Bx = (B * x).contiguous()
            hidden_states = C * Bx
            contextualized_states, _ = self.out_proj(hidden_states)
            return contextualized_states

        num_prefills = attn_metadata.num_prefills  # request count
        num_decodes = attn_metadata.num_decode_tokens  # token count (=request)
        num_prefill_tokens = attn_metadata.num_prefill_tokens  # token count
        has_prefill = num_prefills > 0
        has_decode = num_decodes > 0
        num_actual_tokens = num_decodes + num_prefill_tokens

        # NOTE: V1 puts decode before prefill
        # Separate prefill and decode by splitting varlen input
        # Split along token dimension
        B_d, B_p = torch.split(
            B[:num_actual_tokens],
            [num_decodes, num_prefill_tokens],
            dim=0,
        )
        C_d, C_p = torch.split(
            C[:num_actual_tokens],
            [num_decodes, num_prefill_tokens],
            dim=0,
        )
        x_d, x_p = torch.split(
            x[:num_actual_tokens],
            [num_decodes, num_prefill_tokens],
            dim=0,
        )
        # Split along batch dimension
        state_indices_tensor_d, state_indices_tensor_p = torch.split(
            state_indices_tensor,
            [num_decodes, num_prefills],
            dim=0,
        )
        
        

        if prefix_caching_enabled:
            # If prefix caching is enabled, retrieve the relevant variables
            # for prefill and decode
            # 여기서 프리픽스캐싱된 걸 가져온다 - mamba2_attn에서 가져오는듯
            
            # print("\n이값은 메타데이터의 block_idx_last_computed_token", attn_metadata.block_idx_last_computed_token)
            
            block_idx_last_computed_token_d, block_idx_last_computed_token_p = (
                torch.split(
                    attn_metadata.block_idx_last_computed_token,
                    [num_decodes, num_prefills],
                    dim=0,
                )
            )
            
            # print("\n이값은 메타데이터의 block_idx_last_scheduled_token", attn_metadata.block_idx_last_scheduled_token)
            block_idx_last_scheduled_token_d, block_idx_last_scheduled_token_p = (
                torch.split(
                    attn_metadata.block_idx_last_scheduled_token,
                    [num_decodes, num_prefills],
                    dim=0,
                )
            )
            # Prefill-only variables:
            block_idx_first_scheduled_token_p = (
                attn_metadata.block_idx_first_scheduled_token_p
            )
            
            #print("\n이값은 메타데이터의 block_idx_first_scheduled_token_p", block_idx_first_scheduled_token_p)
            
            num_computed_tokens_p = attn_metadata.num_computed_tokens_p
            
            #print("\n이미계산된토큰", num_computed_tokens_p)
            if num_computed_tokens_p != None and seq_idx_p != None:
                print("\n해당인텍스의 이미계산된토큰: ", num_computed_tokens_p[seq_idx_p])
        else:
            block_idx_last_computed_token_d = None
            block_idx_last_computed_token_p = None
            block_idx_last_scheduled_token_d = None
            block_idx_last_scheduled_token_p = None
            block_idx_first_scheduled_token_p = None
            num_computed_tokens_p = None

        # Preallocate output tensor to avoid memcpy cost for merging prefill
        # and decode outputs
        # preallocated_ssm_out = torch.empty(
        #     [
        #         num_prefill_tokens + num_decodes,
        #         (self.num_heads // self.tp_size) * self.head_dim,
        #     ],
        #     dtype=hidden_states.dtype,
        #     device=hidden_states.device,
        # )
        # preallocated_ssm_out_d, preallocated_ssm_out_p = torch.split(
        #     preallocated_ssm_out,
        #     [num_decodes, num_prefill_tokens],
        #     dim=0,
        # )        
        
        
        #WILL 여기가 먼가 - 이건 attn쪽에서 결국처리해줘서 non-pc때는 동작함
        # query_start_loc_p = (
        #     attn_metadata.query_start_loc[-num_prefills - 1 :] - num_decodes
        #     if has_prefill
        #     else None
        # )

        conv_output_list = []

        print("\n차이가나기직전뽀스트스케줄: ", block_idx_first_scheduled_token_p)
        print("\n!!!이게문제인듯!!!차이가나기직전라스트스케줄: ", block_idx_last_scheduled_token_p)
        
        if has_prefill:
            Bx_p = (B_p * x_p).transpose(0, 1)
            
            #WILL 트리톤용 추가
            state_indices_tensor_p = state_indices_tensor_p.view(-1)
            Bx = causal_conv1d_fn(
                Bx_p,
            #WILL 강제로    
            #     conv_weights,
            #     self.conv.bias,
            #     activation=None,
            #     conv_states=conv_state,
            #     has_initial_state=has_initial_states_p,
            #     cache_indices=state_indices_tensor_p,
            #     metadata=attn_metadata,
            #     query_start_loc=query_start_loc_p,
            # ).transpose(0, 1)[:num_prefill_tokens]
            #WILL 이걸넣으니깐 pc에서 첫번째 출력도 다르게나온다
                conv_weights,
                self.conv.bias, #WILL !!이게다르구나
                activation=None, #WILL 변경
                conv_states=conv_state,
                has_initial_state=has_initial_states_p,
                cache_indices=state_indices_tensor_p,
                #WILL 이 2개를 수정하는까 pc에서 첫번째 출력은 고정되서 나온다. revision- block_idx_last_scheduled_token를 넣으면 첫번째출력이 깨진다
                #block_idx_first_scheduled_token=None,
                block_idx_first_scheduled_token=block_idx_first_scheduled_token_p,
                #block_idx_last_scheduled_token=None,
                block_idx_last_scheduled_token=(block_idx_last_scheduled_token_p if block_idx_last_scheduled_token_p is not None else None),
                initial_state_idx=block_idx_last_computed_token_p, #이게 꽉 찬 블록들 중 마지막 블록의 인덱스다! 
                num_computed_tokens=num_computed_tokens_p,
                block_size_to_align=mamba_block_size, #왜 이건 빼나 넣으나 똑같나. 트리톤에서 BLOCK_M으로 지정한다 BLOCK_M=8이지정. 결국 16하고 8하고 결과는 똑같다. 이건 performance관련.
                metadata=attn_metadata,
                query_start_loc=query_start_loc_p,
                #validate_data=True
            ).transpose(0, 1)[:num_prefill_tokens]         #!!!num_prefill_tokens 이게 실제로 계산되야하는 토큰 갯수를 말하는듯. 즉 마지막블락의 토큰 갯수   

            y = C_p * Bx
            conv_output_list.append(y)
            print("\n컨브아웃풋리스트", conv_output_list)
            
            print(f"\n[DEBUG] {self.prefix} - conv_output_list length:", len(conv_output_list))
            if len(conv_output_list) > 0:
                print(f"[DEBUG] {self.prefix} - first element shape: {tuple(conv_output_list[0].shape)}")
        
            
        # 여기에서 차이가 나느것 같다. 이걸알려면 ssm과의 차이를 알아야하는데, 여기가 문제가 있을수가 없는데. 아니면 커널로 갓 인터피어한다    

        if has_decode:
            print("\n이건 해즈디코딩")       

            # if prefix_caching_enabled:
            #     state_indices_tensor_d_input = state_indices_tensor_d.gather(
            #         1, block_idx_last_computed_token_d.unsqueeze(1)
            #     ).squeeze(1)
            #     state_indices_tensor_d_output = state_indices_tensor_d.gather(
            #         1, block_idx_last_scheduled_token_d.unsqueeze(1)
            #     ).squeeze(1)
            #     # for decode:
            #     #   block_idx_first_scheduled_token_d ==
            #     #       block_idx_last_scheduled_token_d
            #     # at block boundaries:
            #     #   block_idx_first_scheduled_token_d >
            #     #       block_idx_last_computed_token_d
            # else:
            #     # Without caching, read and write in-place to the same blocks:
            #     state_indices_tensor_d_input = state_indices_tensor_d
            #     state_indices_tensor_d_output = state_indices_tensor_d
            
            Bx_d = (B_d * x_d).contiguous()
            Bx = causal_conv1d_update(
                Bx_d,
                conv_state,
                conv_weights,
                self.conv.bias,
                activation=None,
                conv_state_indices=state_indices_tensor_d,
                #WILL 여기추가 - 여기추가한거때문에 pc첫번째가 일치한거였네..위에꺼는 무관하다. 근데 왜 속도가 느려지나
                block_idx_last_scheduled_token=block_idx_last_scheduled_token_d,
                initial_state_idx=block_idx_last_computed_token_d,
                #validate_data=True           
            )
            
            
            y = C_d * Bx
            conv_output_list.insert(0, y)

        # Merge prefill and decode outputs before passing to gated MLP
        hidden_states = torch.vstack(conv_output_list)

        # Final linear projection
        output[:num_actual_tokens], _ = self.out_proj(hidden_states)

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        assert self.model_config is not None
        assert self.cache_config is not None
        return MambaStateDtypeCalculator.short_conv_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, ...]]:
        return MambaStateShapeCalculator.short_conv_state_shape(
            tp_world_size=get_tensor_model_parallel_world_size(),
            intermediate_size=self.conv_dim,
            conv_kernel=self.L_cache,
        )

    @property
    def mamba_type(self) -> str:
        return "short_conv"

    def get_attn_backend(self) -> type["AttentionBackend"]:
        from vllm.v1.attention.backends.short_conv_attn import ShortConvAttentionBackend

        return ShortConvAttentionBackend


def short_conv(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self.forward_cuda(hidden_states=hidden_states, output=output)


def short_conv_fake(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="short_conv",
    op_func=short_conv,
    mutates_args=["output"],
    fake_impl=short_conv_fake,
)
