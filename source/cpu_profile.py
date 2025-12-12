import cProfile
import pstats
import io

from .Model_interface import create_backend
from .food_dataset import load_meals_with_prompts
from .config import BASELINE_MODEL


def profile_single_prompt(
    data_path: str = "data/data.tsv",
    items_path: str = "data/items.tsv",
):
    meals = load_meals_with_prompts(data_path, items_path, max_meals=1)
    prompt = meals[0]

    backend = create_backend(BASELINE_MODEL.name, BASELINE_MODEL.quantization)

    pr = cProfile.Profile()
    pr.enable()
    backend.generate(prompt, max_tokens=BASELINE_MODEL.params_b and 128)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
    ps.print_stats(20)
    print(s.getvalue())


if __name__ == "__main__":
    profile_single_prompt()
