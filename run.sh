
cd VLMEvalKit

export OPENAI_API_KEY=
export LMUData="./LMUData"
export SiliconFlow_API_KEY=


# valid_type: STR, LLM
python -u run.py --data PhyX_mini_IMG \
    --model GPT4o_20241120 \
    --judge deepseek --judge-args '{"valid_type": "LLM"}'
