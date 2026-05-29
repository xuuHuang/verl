import re
from sacrebleu.metrics import CHRF
from verl.utils.reward_score import math_verify

LOW_RESOURCE_LANGS = {"sw", "te"}

def compute_score(
    data_source,
    solution_str,
    ground_truth,
    comet_score,
    llm_judge_score,
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
    use_llm_judge = bool(tapo_config.get("use_llm_judge_in_mixed", False))

    res = 0.0
    chrf_score = 0.0
    math_reward = 0.0
    translation_reward = 0.0
    translation_char_length = 0

    # m = re.search(r"<english_translation>(.*?)</english_translation>", solution_str, re.DOTALL)
    # if m is not None:
    if (translation := extra_info.get("translation", None)) is not None:
        translation_char_length = len(translation) 
        if reference := extra_info.get("en_problem", None) is not None:
            chrf = CHRF(word_order=2)
            chrf_score = round(chrf.sentence_score(translation, [reference]).score / 100.0, 4)
        math_reward = math_verify.compute_score(solution_str, ground_truth)

        # Normalize llm_judge_score from 0-100 to 0-1 to match other scores
        normalized_llm_judge_score = llm_judge_score / 100.0

        match reward_type:
            case "mixed":
                if use_llm_judge:
                    translation_reward = max(round((chrf_score + comet_score + normalized_llm_judge_score) / 3, 4), 0.0)
                else:
                    translation_reward = max(round((chrf_score + comet_score) / 2, 4), 0.0)
            case "llm_judge":
                translation_reward = normalized_llm_judge_score
            case "comet":
                translation_reward = comet_score
            case "chrf++":
                translation_reward = chrf_score
            case "adaptive":
                lang = extra_info.get("lang", None)
                if lang in LOW_RESOURCE_LANGS or lang is None:
                    translation_reward = chrf_score
                else:
                    translation_reward = comet_score

        translation_reward *= lambd

        match aggregate_method:
            case "add":
                res = translation_reward + math_reward
            case "multiplicative":
                res = translation_reward * math_reward
            case "separate":
                res = 0.0
            case _:
                raise NotImplementedError(f"Unsupported aggregate_method: {aggregate_method}")

    mt_score = {}
    if extra_info.get("validate", False):
        mt_score = {"chrf_score": chrf_score}
    else:
        match reward_type:
            case "mixed" | "adaptive":
                if use_llm_judge:
                    mt_score = {"chrf_score": chrf_score, "comet_score": comet_score, "llm_judge_score": llm_judge_score}
                else:
                    mt_score = {"chrf_score": chrf_score, "comet_score": comet_score}
            case "comet":
                mt_score = {"comet_score": comet_score}
            case "chrf++":
                mt_score = {"chrf_score": chrf_score}
            case "llm_judge":
                mt_score = {"llm_judge_score": llm_judge_score}

    return {
        "score": float(res),
        "math_reward": math_reward,
        "translation_reward": translation_reward,
        "translation_char_length": translation_char_length,
        **mt_score,
    }