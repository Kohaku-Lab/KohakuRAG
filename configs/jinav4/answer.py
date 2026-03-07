"""Answer questions using JinaV4 multimodal RAG + OpenRouter.

Usage:
    kogine run scripts/wattbot_answer.py --config configs/jinav4/answer.py
"""

from kohakuengine import Config

# Input/Output
db = "artifacts/wattbot_jinav4.db"
table_prefix = "wattbot_jv4"
questions = "data/test_Q.csv"
output = "artifacts/jinav4_answers.csv"
metadata = "data/metadata.csv"

# LLM settings (using OpenRouter)
llm_provider = "openrouter"  # Options: "openai", "openrouter"
model = "openai/gpt-5-nano"
planner_model = None  # Falls back to model
openrouter_api_key = None  # From env: OPENROUTER_API_KEY
site_url = "https://github.com/KohakuBlueleaf/KohakuRAG"
app_name = "KohakuRAG"

# Retrieval settings
top_k = 16
planner_max_queries = 3
deduplicate_retrieval = True
rerank_strategy = "frequency"  # Options: None, "frequency", "score", "combined"
top_k_final = 24  # Truncate to top-24 after dedup+rerank (None = no truncation)
snippet_dedup = (
    "none"  # Snippet-level dedup after context expansion: "none", "node_id", "tree"
)
parent_depth = 1  # How many parent levels to include during context expansion
child_depth = 0  # How many child levels to include during context expansion

# JinaV4 embedding settings (must match index)
embedding_model = "jinav4"  # Options: "jina" (v3), "jinav4"
embedding_dim = 1024  # Must match index.py. Matryoshka: 128, 256, 512, 1024, 2048
embedding_task = "retrieval"  # Options: "retrieval", "text-matching", "code"

# Paragraph search mode (runtime toggle, requires "both" mode during indexing)
# Options:
#   - "averaged": Use sentence-averaged paragraph embeddings (default)
#   - "full": Use full paragraph embeddings (requires index built with "both" or "full" mode)
paragraph_search_mode = "averaged"

# Image settings (always enabled for JinaV4 multimodal)
with_images = True
top_k_images = 4  # Images from dedicated image search
send_images_to_llm = False  # Send actual image bytes to vision LLM

# Other settings
max_retries = 3
max_concurrent = 10
single_run_debug = False
question_id = None
use_reordered_prompt = False  # Put context before question to combat attention sink


def config_gen():
    return Config.from_globals()
