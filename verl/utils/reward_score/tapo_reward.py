import re
from sacrebleu.metrics import CHRF
from verl.utils.reward_score import math_verify

def compute_score(
    data_source,
    solution_str,
    ground_truth,
    comet_score,
    extra_info=None,
    tapo_config=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    **kwargs,
):
    reward_type = tapo_config["reward_type"]
    aggregate_method = tapo_config["aggregate_method"]
    lambd = tapo_config["lambd"]

    res = 0.0
    chrf_score = 0.0
    math_reward = 0.0
    translation_reward = 0.0

    m = re.search(r"<english_translation>(.*?)</english_translation>", solution_str, re.DOTALL)
    if m is not None:
        reference = extra_info.get("en_problem", None)   
        if reference is not None:
            chrf = CHRF(word_order=2)
            translation = m.group(1).strip()
            chrf_score = chrf.sentence_score(translation, [reference]).score / 100.0
        math_reward = math_verify.compute_score(solution_str, ground_truth)

        match reward_type:
            case "mixed":
                translation_reward = chrf_score + comet_score
            case "comet":
                translation_reward = comet_score
            case "chrf++":
                translation_reward = chrf_score

        translation_reward *= lambd

        match aggregate_method:
            case "add":
                res = translation_reward + math_reward
            case _:
                raise NotImplementedError(f"Unsupported aggregate_method: {aggregate_method}")

    mt_score = {}
    if extra_info.get("validate", False):
        mt_score = {"chrf_score": chrf_score}
    else:
        match reward_type:
            case "mixed":
                mt_score = {"chrf_score": chrf_score, "comet_score": comet_score}
            case "comet":
                mt_score = {"comet_score": comet_score}
            case "chrf++":
                mt_score = {"chrf_score": chrf_score}

    return {
        "score": float(res),
        "math_reward": math_reward,
        "translation_reward": translation_reward,
        **mt_score,
    }