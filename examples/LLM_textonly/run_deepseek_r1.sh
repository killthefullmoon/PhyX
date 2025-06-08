export PHYX_TEXT_ONLY=true

python -u run.py --data PhyX_mini_TL \
    --model deepseek-r1 \
    --judge-args '{"valid_type": "STR"}'


export SiliconFlow_API_KEY=$Your_key 
python -u run.py --data PhyX_mini_TL \
    --model deepseek-r1 \
    --judge deepseek-v3-si --judge-args '{"valid_type": "LLM"}'


export Deepseek_API=$Your_key
python -u run.py --data PhyX_mini_TL \
    --model deepseek-r1 \
    --judge deepseek-v3 --judge-args '{"valid_type": "LLM"}'