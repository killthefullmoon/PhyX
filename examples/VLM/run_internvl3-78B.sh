export  Intern_API_KEY=$Your_key


python -u run.py --data PhyX_mini \
    --model InternVL3-78B-API \
    --judge-args '{"valid_type": "STR"}'


export SiliconFlow_API_KEY=$Your_key
python -u run.py --data PhyX_mini \
    --model InternVL3-78B-API \
    --judge deepseek-v3-si --judge-args '{"valid_type": "LLM"}'


export Deepseek_API=$Your_key
export OPENAI_API_BASE="https://api.deepseek.com/v1/chat/completions"
python -u run.py --data PhyX_mini \
    --model InternVL3-78B-API \
    --judge deepseek-v3 --judge-args '{"valid_type": "LLM"}'
