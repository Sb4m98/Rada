import uuid
import os
import json
from collections import defaultdict

class LocalPersonalizer:
    def __init__(self, storage_path="personalizer_data.json"):
        self.history = []  # Lista di eventi
        self.storage_path = storage_path
        self._load()

    def _load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
            except Exception as e:
                print(f"⚠️ Failed to load personalizer history: {e}")

    def _save(self):
        try:
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to save personalizer history: {e}")

    def send_reward(self, event_id: str, reward: float):
        for entry in self.history:
            if entry["event_id"] == event_id:
                entry["reward"] = reward
                self._save()
                return
        print(f"⚠️ Event ID {event_id} not found for reward assignment")

    def get_best_style_for_profile(self, profile: dict) -> str:
        user_id = profile.get("user_id")
        if not user_id:
            return None

        style_scores = defaultdict(list)
        for h in self.history:
            if h["profile"].get("user_id") == user_id and "reward" in h:
                style = h["chosen"]
                style_scores[style].append(h["reward"])

        avg_scores = {
            style: sum(scores)/len(scores) for style, scores in style_scores.items() if scores
        }

        return max(avg_scores, key=avg_scores.get) if avg_scores else None

    def rank(self, profile: dict, candidates: dict):
        event_id = str(uuid.uuid4())

        best_style = self.get_best_style_for_profile(profile)
        if best_style and best_style in candidates:
            chosen = best_style
        else:
            # fallback su stile preferito se esiste
            chosen = profile.get("preferred_style", "formale")
            if chosen not in candidates:
                chosen = next(iter(candidates))  # fallback qualsiasi

        self.history.append({
            "event_id": event_id,
            "profile": profile,
            "chosen": chosen,
            "candidates": list(candidates.keys()),
            "reward": None
        })
        self._save()
        return chosen, candidates[chosen], event_id
