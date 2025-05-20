import pyautogui
import time
import json
import os

class ComputerController:
    def __init__(self, config_path='utils/config.json'):
        self.last_action_time = 0
        self.cooldown = 1.0  # seconds between actions
        self.actions = self.load_config(config_path)

    def load_config(self, path):
        if not os.path.exists(path):
            print(f"[❌] Config file not found at {path}")
            return {}
        with open(path, 'r') as f:
            return json.load(f)

    def can_trigger(self):
        return time.time() - self.last_action_time > self.cooldown

    def update_last_action_time(self):
        self.last_action_time = time.time()

    def perform_action(self, label):
        if not self.can_trigger():
            return
        action = self.actions.get(label)
        if not action:
            print(f"[⚠️] No action mapped for label '{label}'")
            return

        print(f"[🖥️] Performing action: {action}")
        self.execute_action(action)
        self.update_last_action_time()

    def execute_action(self, action):
        if len(action) == 1:
            pyautogui.press(action[0])
        elif len(action) == 2:
            pyautogui.hotkey(action[0], action[1])
        elif len(action) == 3: # just for shaking hand action
            pyautogui.hotkey(action[0], action[1])
            pyautogui.press(action[2])

if __name__ == "__main__":
    controller = ComputerController()
    # Example usage
    print(controller.actions.get("Shaking Hand"))

