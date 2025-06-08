
python -u run.py --data PhyX_mini \
    --model GPT4o_20241120 \
    --judge-args '{"valid_type": "STR"}'


export SiliconFlow_API_KEY=$Your_key 
python -u run.py --data PhyX_mini \
    --model GPT4o_20241120 \
    --judge deepseek-v3-si --judge-args '{"valid_type": "LLM"}'


export Deepseek_API=$Your_key
python -u run.py --data PhyX_mini \
    --model GPT4o_20241120 \
    --judge deepseek-v3 --judge-args '{"valid_type": "LLM"}'
    