# Copyright 2025 xuuHuang. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import re
from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


TRANSLATION_TAG_PATTERN = re.compile(r"<english_translation>(.*?)</english_translation>", re.DOTALL)


def _extract_english_translation(solution_str: str) -> str | None:
    match = TRANSLATION_TAG_PATTERN.search(solution_str)
    return None if match is None else match.group(1).strip()


JUDGE_PROMPT_TEMPLATE = (
    "Score the following translation from {source_lang} to {target_lang} "
    "on a scale from 0 to 100, where a score of 0 means a broken or poor translation; "
    "33 indicates a flawed translation with significant issues; "
    "66 indicates a good translation with only minor issues in grammar, fluency, or consistency; "
    "and 100 represents a perfect translation in both meaning and grammar.\n\n"
    "Answer with only a whole number representing the score, and nothing else.\n\n"
    "{source_lang} source text:\n{source_seg}\n"
    "{target_lang} translation:\n{target_seg}"
)

LANGUAGE_NAME_MAPPING = {
    "bn": "Bengali",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "ja": "Japanese",
    "ru": "Russian",
    "sw": "Swahili",
    "te": "Telugu",
    "th": "Thai",
    "zh": "Chinese",
}

@register("tapo")
class TapoRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        tapo_config=None
    ) -> None:
        """
        Initialize the TapoRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        self.tapo_config = tapo_config
        self.llm_judge_config = (self.tapo_config or {}).get("llm_judge", {})

    def _llm_judge_enabled(self) -> bool:
        return bool(self.llm_judge_config.get("enable", False))

    def _extract_numeric_judge_score(self, judge_text: str) -> float:
        numeric_tokens = re.findall(r"-?\d+(?:\.\d+)?", judge_text)
        if not numeric_tokens:
            return 0.0

        try:
            candidate_scores = [float(token) for token in numeric_tokens]
        except ValueError:
            return 0.0

        min_score = float(self.llm_judge_config.get("min_score", 0.0))
        max_score = float(self.llm_judge_config.get("max_score", 100.0))
        for score in candidate_scores:
            if min_score <= score <= max_score:
                return score

        return min(max(candidate_scores[0], 0.0), 100.0)

    def _build_judge_prompt(self, source_problem: str, source_language: str, translated_english: str, reference_english: str) -> str:
        return JUDGE_PROMPT_TEMPLATE.format(
            source_lang=source_language,
            target_lang="English",
            source_seg=source_problem,
            target_seg=translated_english
        )

    async def _judge_single_translation_async(self, client, sem: asyncio.Semaphore, judge_item: dict[str, str]) -> float:
        translation = judge_item["candidate_translation"]
        reference = judge_item["reference_translation"]

        if not translation or not reference:
            return 0.0

        model = self.llm_judge_config.get("model", None)
        if not model:
            return 0.0

        max_retries = int(self.llm_judge_config.get("max_retries", 2))
        request_timeout = float(self.llm_judge_config.get("request_timeout", 30.0))
        temperature = float(self.llm_judge_config.get("temperature", 0.0))

        prompt = self._build_judge_prompt(
            source_language=LANGUAGE_NAME_MAPPING.get(judge_item["source_language"], judge_item["source_language"]),
            source_problem=judge_item["source_problem"],
            translated_english=translation,
            reference_english=reference,
        )

        for attempt in range(max_retries + 1):
            try:
                async with sem:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        timeout=request_timeout,
                    )
                judge_text = response.choices[0].message.content.strip()
                parsed_score = self._extract_numeric_judge_score(judge_text)
                return parsed_score
            except Exception as error:
                if attempt >= max_retries:
                    print(f"[tapo][llm_judge] failed after {max_retries + 1} attempts: {error}")
                    return 0.0
                await asyncio.sleep(min(2**attempt, 3))

        return 0.0

    async def _evaluate_translation_batch_async(self, judge_inputs: list[dict[str, str]]) -> list[float]:
        if not judge_inputs:
            return []

        try:
            from openai import AsyncOpenAI
        except Exception as error:
            print(f"[tapo][llm_judge] openai package is unavailable: {error}")
            return [0.0] * len(judge_inputs)

        base_url = self.llm_judge_config.get("base_url", None)
        api_key = self.llm_judge_config.get("api_key", "EMPTY")
        max_concurrency = max(1, int(self.llm_judge_config.get("max_concurrency", 16)))

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        client = AsyncOpenAI(**client_kwargs)
        sem = asyncio.Semaphore(max_concurrency)

        tasks = [self._judge_single_translation_async(client, sem, judge_item) for judge_item in judge_inputs]
        scores = await asyncio.gather(*tasks, return_exceptions=True)

        final_scores: list[float] = [
            0.0 if isinstance(score, Exception) else float(score) for score in scores
        ]
        for score in scores:
            if isinstance(score, Exception):
                print(f"[tapo][llm_judge] async task error: {score}")

        await client.close()
        return final_scores

    def _evaluate_translation_batch(self, judge_inputs: list[dict[str, str]]) -> list[float]:
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop and running_loop.is_running():
            print("[tapo][llm_judge] running loop detected; skip llm_judge for this batch to avoid nested event loop")
            return [0.0] * len(judge_inputs)
        return asyncio.run(self._evaluate_translation_batch_async(judge_inputs))

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""

        if "comet_score" in data.batch.keys():
            comet_score = data.batch["comet_score"]
        else:
            comet_score = torch.zeros(len(data), dtype=torch.float32)

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        # if "rm_scores" in data.batch.keys():
        #     if return_dict:
        #         reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
        #         reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
        #         return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
        #     else:
        #         return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}
        batch_items: list[dict[str, Any]] = []
        judge_inputs: list[dict[str, str]] = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["num_turns"] = num_turns
            extra_info["rollout_reward_scores"] = rollout_reward_scores

            translation = _extract_english_translation(response_str)
            extra_info["translation"] = translation
            judge_inputs.append(
                {
                    "source_problem": extra_info["src_problem"],
                    "source_language": extra_info["lang"],
                    "candidate_translation": translation,
                    "reference_translation": extra_info["en_problem"],
                }
            )

            batch_items.append(
                {
                    "prompt_str": prompt_str,
                    "response_str": response_str,
                    "ground_truth": ground_truth,
                    "data_source": data_source,
                    "extra_info": extra_info,
                    "valid_response_length": valid_response_length,
                }
            )

        llm_judge_scores = (
            self._evaluate_translation_batch(judge_inputs)
            if self._llm_judge_enabled()
            else [0.0] * len(batch_items)
        )
        assert len(llm_judge_scores) == len(batch_items), "Length of llm_judge_scores must match the number of batch items"

        for i, item in enumerate(batch_items):

            prompt_str = item["prompt_str"]
            response_str = item["response_str"]
            ground_truth = item["ground_truth"]
            data_source = item["data_source"]
            extra_info = item["extra_info"]
            valid_response_length = item["valid_response_length"]

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                comet_score=comet_score[i].item(),
                llm_judge_score=llm_judge_scores[i],
                extra_info=extra_info,
                tapo_config=self.tapo_config,
            )

            if self.tapo_config["aggregate_method"] == "separate":
                translation_char_length = score.pop("translation_char_length")
                _ = score.pop("score")
                translation_token_length = 0
                if translation_char_length > 0:
                    translation_tokens = self.tokenizer.encode(response_str[:translation_char_length], add_special_tokens=False)
                    translation_token_length = len(translation_tokens)
                if translation_token_length < valid_response_length:
                    reward_tensor[i, :translation_token_length] = score["translation_reward"]
                    reward_tensor[i, translation_token_length:] = score["math_reward"]
                else:
                    pass
                # data_item.non_tensor_batch["translation_token_length"] = translation_token_length
                score["translation_token_length"] = translation_token_length
            else:
                reward = score["score"]
                reward_tensor[i, valid_response_length - 1] = reward

            # Store the information including original reward
            for key, value in score.items():
                reward_extra_info[key].append(value)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
