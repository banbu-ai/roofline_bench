# build llama.cpp

https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md

# llama-bench

https://github.com/ggml-org/llama.cpp/tree/master/tools/llama-

```bash
usage: llama-bench [options]

options:
  -h, --help
  --numa <distribute|isolate|numactl>       numa mode (default: disabled)
  -r, --repetitions <n>                     number of times to repeat each test (default: 5)
  --prio <0|1|2|3>                          process/thread priority (default: 0)
  --delay <0...N> (seconds)                 delay between each test (default: 0)
  -o, --output <csv|json|jsonl|md|sql>      output format printed to stdout (default: md)
  -oe, --output-err <csv|json|jsonl|md|sql> output format printed to stderr (default: none)
  -v, --verbose                             verbose output
  --progress                                print test progress indicators

test parameters:
  -m, --model <filename>                    (default: models/7B/ggml-model-q4_0.gguf)
  -p, --n-prompt <n>                        (default: 512)
  -n, --n-gen <n>                           (default: 128)
  -pg <pp,tg>                               (default: )
  -d, --n-depth <n>                         (default: 0)
  -b, --batch-size <n>                      (default: 2048)
  -ub, --ubatch-size <n>                    (default: 512)
  -ctk, --cache-type-k <t>                  (default: f16)
  -ctv, --cache-type-v <t>                  (default: f16)
  -dt, --defrag-thold <f>                   (default: -1)
  -t, --threads <n>                         (default: system dependent)
  -C, --cpu-mask <hex,hex>                  (default: 0x0)
  --cpu-strict <0|1>                        (default: 0)
  --poll <0...100>                          (default: 50)
  -ngl, --n-gpu-layers <n>                  (default: 99)
  -rpc, --rpc <rpc_servers>                 (default: none)
  -sm, --split-mode <none|layer|row>        (default: layer)
  -mg, --main-gpu <i>                       (default: 0)
  -nkvo, --no-kv-offload <0|1>              (default: 0)
  -fa, --flash-attn <0|1>                   (default: 0)
  -mmp, --mmap <0|1>                        (default: 1)
  -embd, --embeddings <0|1>                 (default: 0)
  -ts, --tensor-split <ts0/ts1/..>          (default: 0)
  -ot --override-tensors <tensor name pattern>=<buffer type>;...
                                            (default: disabled)
  -nopo, --no-op-offload <0|1>              (default: 0)

Multiple values can be given for each parameter by separating them with ','
or by specifying the parameter multiple times. Ranges can be given as
'first-last' or 'first-last+step' or 'first-last*mult'.
```

# convert model to gguf

convert_hf_to_gguf.py help
```bash
usage: convert_hf_to_gguf.py [-h] [--vocab-only] [--outfile OUTFILE] [--outtype {f32,f16,bf16,q8_0,tq1_0,tq2_0,auto}] [--bigendian] [--use-temp-file] [--no-lazy] [--model-name MODEL_NAME] [--verbose]
                             [--split-max-tensors SPLIT_MAX_TENSORS] [--split-max-size SPLIT_MAX_SIZE] [--dry-run] [--no-tensor-first-split] [--metadata METADATA] [--print-supported-models] [--remote] [--mmproj]
                             [--mistral-format]
                             [model]

Convert a huggingface model to a GGML compatible file

positional arguments:
  model                 directory containing model file or huggingface repository ID (if --remote)

options:
  -h, --help            show this help message and exit
  --vocab-only          extract only the vocab
  --outfile OUTFILE     path to write to; default: based on input. {ftype} will be replaced by the outtype.
  --outtype {f32,f16,bf16,q8_0,tq1_0,tq2_0,auto}
                        output format - use f32 for float32, f16 for float16, bf16 for bfloat16, q8_0 for Q8_0, tq1_0 or tq2_0 for ternary, and auto for the highest-fidelity 16-bit float type depending on the
                        first loaded tensor type
  --bigendian           model is executed on big endian machine
  --use-temp-file       use the tempfile library while processing (helpful when running out of memory, process killed)
  --no-lazy             use more RAM by computing all outputs before writing (use in case lazy evaluation is broken)
  --model-name MODEL_NAME
                        name of the model
  --verbose             increase output verbosity
  --split-max-tensors SPLIT_MAX_TENSORS
                        max tensors in each split
  --split-max-size SPLIT_MAX_SIZE
                        max size per split N(M|G)
  --dry-run             only print out a split plan and exit, without writing any new files
  --no-tensor-first-split
                        do not add tensors to the first split (disabled by default)
  --metadata METADATA   Specify the path for an authorship metadata override file
  --print-supported-models
                        Print the supported models
  --remote              (Experimental) Read safetensors file remotely without downloading to disk. Config and tokenizer files will still be downloaded. To use this feature, you need to specify Hugging Face model
                        repo name instead of a local directory. For example: 'HuggingFaceTB/SmolLM2-1.7B-Instruct'. Note: To access gated repo, set HF_TOKEN environment variable to your Hugging Face token.
  --mmproj              (Experimental) Export multimodal projector (mmproj) for vision models. This will only work on some vision models. A prefix 'mmproj-' will be added to the output file name.
  --mistral-format      Whether the model is stored following the Mistral format.
```

# Supported convert models

```bash
(llama.cpp) ➜  llama.cpp git:(master) python convert_hf_to_gguf.py --print-supported-models
Supported models:
TEXT models:
  - ArceeForCausalLM
  - ArcticForCausalLM
  - BaiChuanForCausalLM
  - BaichuanForCausalLM
  - BailingMoeForCausalLM
  - BambaForCausalLM
  - BertForMaskedLM
  - BertForSequenceClassification
  - BertModel
  - BitnetForCausalLM
  - BloomForCausalLM
  - BloomModel
  - CamembertModel
  - ChameleonForCausalLM
  - ChameleonForConditionalGeneration
  - ChatGLMForConditionalGeneration
  - ChatGLMModel
  - CodeShellForCausalLM
  - Cohere2ForCausalLM
  - CohereForCausalLM
  - DbrxForCausalLM
  - DeciLMForCausalLM
  - DeepseekForCausalLM
  - DeepseekV2ForCausalLM
  - DeepseekV3ForCausalLM
  - DistilBertForMaskedLM
  - DistilBertForSequenceClassification
  - DistilBertModel
  - Dots1ForCausalLM
  - DreamModel
  - Ernie4_5_ForCausalLM
  - Ernie4_5_MoeForCausalLM
  - Exaone4ForCausalLM
  - ExaoneForCausalLM
  - FalconForCausalLM
  - FalconH1ForCausalLM
  - FalconMambaForCausalLM
  - GPT2LMHeadModel
  - GPTBigCodeForCausalLM
  - GPTNeoXForCausalLM
  - GPTRefactForCausalLM
  - Gemma2ForCausalLM
  - Gemma3ForCausalLM
  - Gemma3ForConditionalGeneration
  - Gemma3nForConditionalGeneration
  - GemmaForCausalLM
  - Glm4ForCausalLM
  - Glm4MoeForCausalLM
  - Glm4vForConditionalGeneration
  - GlmForCausalLM
  - GptOssForCausalLM
  - GraniteForCausalLM
  - GraniteMoeForCausalLM
  - GraniteMoeHybridForCausalLM
  - GraniteMoeSharedForCausalLM
  - GrokForCausalLM
  - HunYuanDenseV1ForCausalLM
  - HunYuanMoEV1ForCausalLM
  - InternLM2ForCausalLM
  - InternLM3ForCausalLM
  - JAISLMHeadModel
  - JambaForCausalLM
  - JinaBertForMaskedLM
  - JinaBertModel
  - KimiVLForConditionalGeneration
  - LFM2ForCausalLM
  - LLaDAModelLM
  - LLaMAForCausalLM
  - Lfm2ForCausalLM
  - Llama4ForConditionalGeneration
  - LlamaForCausalLM
  - LlamaModel
  - LlavaForConditionalGeneration
  - LlavaStableLMEpochForCausalLM
  - MPTForCausalLM
  - MT5ForConditionalGeneration
  - Mamba2ForCausalLM
  - MambaForCausalLM
  - MambaLMHeadModel
  - MiniCPM3ForCausalLM
  - MiniCPMForCausalLM
  - Mistral3ForConditionalGeneration
  - MistralForCausalLM
  - MixtralForCausalLM
  - NemotronForCausalLM
  - NeoBERT
  - NeoBERTForSequenceClassification
  - NeoBERTLMHead
  - NomicBertModel
  - OLMoForCausalLM
  - Olmo2ForCausalLM
  - OlmoForCausalLM
  - OlmoeForCausalLM
  - OpenELMForCausalLM
  - OrionForCausalLM
  - PLMForCausalLM
  - PLaMo2ForCausalLM
  - Phi3ForCausalLM
  - PhiForCausalLM
  - PhiMoEForCausalLM
  - Plamo2ForCausalLM
  - PlamoForCausalLM
  - QWenLMHeadModel
  - Qwen2AudioForConditionalGeneration
  - Qwen2ForCausalLM
  - Qwen2Model
  - Qwen2MoeForCausalLM
  - Qwen2VLForConditionalGeneration
  - Qwen2VLModel
  - Qwen2_5OmniModel
  - Qwen2_5_VLForConditionalGeneration
  - Qwen3ForCausalLM
  - Qwen3MoeForCausalLM
  - RWForCausalLM
  - RWKV6Qwen2ForCausalLM
  - RWKV7ForCausalLM
  - RobertaForSequenceClassification
  - RobertaModel
  - Rwkv6ForCausalLM
  - Rwkv7ForCausalLM
  - RwkvHybridForCausalLM
  - SmallThinkerForCausalLM
  - SmolLM3ForCausalLM
  - StableLMEpochForCausalLM
  - StableLmForCausalLM
  - Starcoder2ForCausalLM
  - T5EncoderModel
  - T5ForConditionalGeneration
  - T5WithLMHeadModel
  - UMT5ForConditionalGeneration
  - UltravoxModel
  - VLlama3ForCausalLM
  - VoxtralForConditionalGeneration
  - WavTokenizerDec
  - XLMRobertaForSequenceClassification
  - XLMRobertaModel
  - XverseForCausalLM
MMPROJ models:
  - Gemma3ForConditionalGeneration
  - Idefics3ForConditionalGeneration
  - InternVisionModel
  - Llama4ForConditionalGeneration
  - LlavaForConditionalGeneration
  - Mistral3ForConditionalGeneration
  - Qwen2AudioForConditionalGeneration
  - Qwen2VLForConditionalGeneration
  - Qwen2VLModel
  - Qwen2_5OmniModel
  - Qwen2_5_VLForConditionalGeneration
  - SmolVLMForConditionalGeneration
  - UltravoxModel
  - VoxtralForConditionalGeneration
```

# llama-quantize

```bash
usage: llama-quantize [--help] [--allow-requantize] [--leave-output-tensor] [--pure] [--imatrix] [--include-weights]
       [--exclude-weights] [--output-tensor-type] [--token-embedding-type] [--tensor-type] [--prune-layers] [--keep-split] [--override-kv]
       model-f32.gguf [model-quant.gguf] type [nthreads]

  --allow-requantize: Allows requantizing tensors that have already been quantized. Warning: This can severely reduce quality compared to quantizing from 16bit or 32bit
  --leave-output-tensor: Will leave output.weight un(re)quantized. Increases model size but may also increase quality, especially when requantizing
  --pure: Disable k-quant mixtures and quantize all tensors to the same type
  --imatrix file_name: use data in file_name as importance matrix for quant optimizations
  --include-weights tensor_name: use importance matrix for this/these tensor(s)
  --exclude-weights tensor_name: use importance matrix for this/these tensor(s)
  --output-tensor-type ggml_type: use this ggml_type for the output.weight tensor
  --token-embedding-type ggml_type: use this ggml_type for the token embeddings tensor
  --tensor-type TENSOR=TYPE: quantize this tensor to this ggml_type. example: --tensor-type attn_q=q8_0
      Advanced option to selectively quantize tensors. May be specified multiple times.
  --prune-layers L0,L1,L2...comma-separated list of layer numbers to prune from the model
      Advanced option to remove all tensors from the given layers
  --keep-split: will generate quantized model in the same shards as input
  --override-kv KEY=TYPE:VALUE
      Advanced option to override model metadata by key in the quantized model. May be specified multiple times.
Note: --include-weights and --exclude-weights cannot be used together

Allowed quantization types:
   2  or  Q4_0    :  4.34G, +0.4685 ppl @ Llama-3-8B
   3  or  Q4_1    :  4.78G, +0.4511 ppl @ Llama-3-8B
  38  or  MXFP4_MOE :  MXFP4 MoE
   8  or  Q5_0    :  5.21G, +0.1316 ppl @ Llama-3-8B
   9  or  Q5_1    :  5.65G, +0.1062 ppl @ Llama-3-8B
  19  or  IQ2_XXS :  2.06 bpw quantization
  20  or  IQ2_XS  :  2.31 bpw quantization
  28  or  IQ2_S   :  2.5  bpw quantization
  29  or  IQ2_M   :  2.7  bpw quantization
  24  or  IQ1_S   :  1.56 bpw quantization
  31  or  IQ1_M   :  1.75 bpw quantization
  36  or  TQ1_0   :  1.69 bpw ternarization
  37  or  TQ2_0   :  2.06 bpw ternarization
  10  or  Q2_K    :  2.96G, +3.5199 ppl @ Llama-3-8B
  21  or  Q2_K_S  :  2.96G, +3.1836 ppl @ Llama-3-8B
  23  or  IQ3_XXS :  3.06 bpw quantization
  26  or  IQ3_S   :  3.44 bpw quantization
  27  or  IQ3_M   :  3.66 bpw quantization mix
  12  or  Q3_K    : alias for Q3_K_M
  22  or  IQ3_XS  :  3.3 bpw quantization
  11  or  Q3_K_S  :  3.41G, +1.6321 ppl @ Llama-3-8B
  12  or  Q3_K_M  :  3.74G, +0.6569 ppl @ Llama-3-8B
  13  or  Q3_K_L  :  4.03G, +0.5562 ppl @ Llama-3-8B
  25  or  IQ4_NL  :  4.50 bpw non-linear quantization
  30  or  IQ4_XS  :  4.25 bpw non-linear quantization
  15  or  Q4_K    : alias for Q4_K_M
  14  or  Q4_K_S  :  4.37G, +0.2689 ppl @ Llama-3-8B
  15  or  Q4_K_M  :  4.58G, +0.1754 ppl @ Llama-3-8B
  17  or  Q5_K    : alias for Q5_K_M
  16  or  Q5_K_S  :  5.21G, +0.1049 ppl @ Llama-3-8B
  17  or  Q5_K_M  :  5.33G, +0.0569 ppl @ Llama-3-8B
  18  or  Q6_K    :  6.14G, +0.0217 ppl @ Llama-3-8B
   7  or  Q8_0    :  7.96G, +0.0026 ppl @ Llama-3-8B
   1  or  F16     : 14.00G, +0.0020 ppl @ Mistral-7B
  32  or  BF16    : 14.00G, -0.0050 ppl @ Mistral-7B
   0  or  F32     : 26.00G              @ 7B
          COPY    : only copy tensors, no quantizing
```

