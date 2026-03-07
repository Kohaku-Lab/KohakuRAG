"""Config for wattbot_answer.py (with-images path)

Usage:
    kogine run scripts/wattbot_answer.py --config configs/with_images/answer.py
"""

from kohakuengine import Config

# Input/Output
db = "artifacts/wattbot_with_images.db"
table_prefix = "wattbot_img"
questions = "data/train_QA.csv"
output = "artifacts/with_images_train_preds3.csv"
metadata = "data/metadata.csv"

# LLM settings
llm_provider = "openrouter"  # Options: "openai", "openrouter"
model = "openai/GPT-5-mini"
planner_model = None  # Falls back to model
openrouter_api_key = None  # From env: OPENROUTER_API_KEY
site_url = "https://github.com/KohakuBlueleaf/KohakuRAG"
app_name = "KohakuRAG"

# Retrieval settings
top_k = 16
planner_max_queries = 3
deduplicate_retrieval = True  # Deduplicate text results by node_id across queries
rerank_strategy = "combined"  # Options: None, "frequency", "score", "combined"
top_k_final = (
    24  # Truncate to this many results after dedup+rerank (None = no truncation)
)
snippet_dedup = (
    "none"  # Snippet-level dedup after context expansion: "none", "node_id", "tree"
)
parent_depth = 1  # How many parent levels to include during context expansion
child_depth = 0  # How many child levels to include during context expansion

# Embedding settings (must match index)
embedding_model = "jina"  # Options: "jina" (v3), "jinav4"
embedding_dim = None  # Only for jinav4: 128, 256, 512, 1024, 2048
embedding_task = "retrieval"  # Options: "retrieval", "text-matching", "code"

# Paragraph search mode (runtime toggle, requires "both" mode during indexing)
# Options:
#   - "averaged": Use sentence-averaged paragraph embeddings (default)
#   - "full": Use full paragraph embeddings (requires index built with "both" or "full" mode)
paragraph_search_mode = "averaged"

# Image settings
with_images = True
top_k_images = 4  # Images from dedicated image search
send_images_to_llm = False  # Send actual image bytes to vision LLM

# Other settings
max_retries = 3
max_concurrent = -1
single_run_debug = False
question_id = None
use_reordered_prompt = False  # Put context before question to combat attention sink


def config_gen():
    return Config.from_globals()
