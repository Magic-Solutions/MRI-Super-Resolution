import time
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pygame
import torch
from PIL import Image

from csgo.action_processing import CSGOAction
from depth_viz import colorize_inverse_depth_uint8
from .dataset_env import DatasetEnv
from .play_env import PlayEnv

_LOGO_PATH = Path(__file__).resolve().parents[2] / "omgrab_logo.png"


class Game:
    def __init__(
        self,
        play_env: Union[PlayEnv, DatasetEnv],
        size: Tuple[int, int],
        mouse_multiplier: int,
        fps: int,
        verbose: bool,
    ) -> None:
        self.env = play_env
        self.height, self.width = size
        self.mouse_multiplier = mouse_multiplier
        self.fps = fps
        self.verbose = verbose

        self.env.print_controls()
        print("\nControls:\n")
        print(" m  : switch control (human/replay)") # Not for main as Game can use either PlayEnv or DatasetEnv
        print(" .  : pause/unpause")
        print(" e  : step-by-step (when paused)")
        print(" ⏎  : reset env")
        print("Esc : quit")
        print("\n")
        input("Press enter to start")

    def run(self) -> None:
        pygame.init()

        screen = pygame.display.set_mode((1280, 720))
        logo_surface = None
        if _LOGO_PATH.exists():
            logo_img = pygame.image.load(str(_LOGO_PATH))
            logo_h = 48
            logo_w = int(logo_img.get_width() * logo_h / logo_img.get_height())
            logo_surface = pygame.transform.smoothscale(logo_img, (logo_w, logo_h))
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)
        clock = pygame.time.Clock()
        label_font = pygame.font.SysFont("mono", 14)
        x_center, y_center = screen.get_rect().center

        device = getattr(getattr(self.env, "agent", None), "device", None)
        if device is None:
            device_str = "cpu"
        elif isinstance(device, torch.device):
            device_str = device.type.upper()
            if device.type == "cuda":
                device_str = torch.cuda.get_device_name(device.index or 0)
            elif device.type == "mps":
                device_str = "Apple MPS"
        else:
            device_str = str(device).upper()

        has_depth = True  # RGBD model outputs 4 channels
        gap = 10  # pixel gap between RGB and depth panels
        prev_time = time.monotonic()
        current_fps = 0.0

        key_size = 36
        key_gap = 4
        color_active = pygame.Color(80, 200, 255)
        color_inactive = pygame.Color(50, 50, 50)
        color_border = pygame.Color(100, 100, 100)
        color_text_active = pygame.Color(0, 0, 0)
        color_text_inactive = pygame.Color(120, 120, 120)
        grab_font = pygame.font.SysFont("mono", 14, bold=True)

        def draw_arrow_key(surface, x, y, w, h, direction, active):
            color = color_active if active else color_inactive
            txt_color = color_text_active if active else color_text_inactive
            pygame.draw.rect(surface, color, (x, y, w, h), border_radius=4)
            pygame.draw.rect(surface, color_border, (x, y, w, h), 1, border_radius=4)
            cx, cy = x + w // 2, y + h // 2
            s = 8
            if direction == "up":
                pts = [(cx, cy - s), (cx - s, cy + s), (cx + s, cy + s)]
            elif direction == "down":
                pts = [(cx, cy + s), (cx - s, cy - s), (cx + s, cy - s)]
            elif direction == "left":
                pts = [(cx - s, cy), (cx + s, cy - s), (cx + s, cy + s)]
            elif direction == "right":
                pts = [(cx + s, cy), (cx - s, cy - s), (cx - s, cy + s)]
            pygame.draw.polygon(surface, txt_color, pts)

        def draw_action_indicator(csgo_action):
            action_names = set()
            for k in csgo_action.keys:
                name = pygame.key.name(k)
                action_names.add(name)

            up = "w" in action_names
            down = "s" in action_names
            left = "a" in action_names
            right = "d" in action_names
            grab = "space" in action_names

            arrows_w = key_size * 3 + key_gap * 2
            grab_w = key_size * 3 + key_gap * 2
            total_w = arrows_w + 40 + grab_w
            hud_x = x_center - total_w // 2
            hud_y = y_center + self.height // 2 + 44

            ax = hud_x
            draw_arrow_key(screen, ax + key_size + key_gap, hud_y, key_size, key_size, "up", up)
            draw_arrow_key(screen, ax, hud_y + key_size + key_gap, key_size, key_size, "left", left)
            draw_arrow_key(screen, ax + key_size + key_gap, hud_y + key_size + key_gap, key_size, key_size, "down", down)
            draw_arrow_key(screen, ax + 2 * (key_size + key_gap), hud_y + key_size + key_gap, key_size, key_size, "right", right)

            gx = hud_x + arrows_w + 40
            gy = hud_y + key_size + key_gap
            gw = grab_w
            gh = key_size
            g_color = color_active if grab else color_inactive
            g_txt_color = color_text_active if grab else color_text_inactive
            pygame.draw.rect(screen, g_color, (gx, gy, gw, gh), border_radius=4)
            pygame.draw.rect(screen, color_border, (gx, gy, gw, gh), 1, border_radius=4)
            txt = grab_font.render("GRAB", True, g_txt_color)
            screen.blit(txt, (gx + (gw - txt.get_width()) // 2, gy + (gh - txt.get_height()) // 2))

        def draw_obs(obs):
            assert obs.ndim == 4 and obs.size(0) == 1
            rgb = obs[0, :3]
            rgb_np = rgb.add(1).div(2).mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            img = Image.fromarray(rgb_np)
            pygame_image = np.array(img.resize((self.width, self.height), resample=Image.BICUBIC)).transpose((1, 0, 2))
            surface = pygame.surfarray.make_surface(pygame_image)

            if has_depth and obs.size(1) >= 4:
                x_rgb = x_center - self.width - gap // 2
                screen.blit(surface, (x_rgb, y_center - self.height // 2))

                depth = obs[0, 3]
                depth_u8 = depth.add(1).div(2).mul(255).clamp(0, 255).byte().cpu().numpy()
                depth_rgb = colorize_inverse_depth_uint8(depth_u8)
                depth_img = Image.fromarray(depth_rgb)
                depth_pygame = np.array(depth_img.resize((self.width, self.height), resample=Image.NEAREST)).transpose((1, 0, 2))
                depth_surface = pygame.surfarray.make_surface(depth_pygame)
                x_depth = x_center + gap // 2
                screen.blit(depth_surface, (x_depth, y_center - self.height // 2))

                rgb_label = label_font.render("RGB", True, pygame.Color(180, 180, 180))
                depth_label = label_font.render("DEPTH", True, pygame.Color(180, 180, 180))
                label_y = y_center - self.height // 2 - 20
                screen.blit(rgb_label, (x_rgb + self.width // 2 - rgb_label.get_width() // 2, label_y))
                screen.blit(depth_label, (x_depth + self.width // 2 - depth_label.get_width() // 2, label_y))
            else:
                screen.blit(surface, (x_center - self.width // 2, y_center - self.height // 2))

        def reset():
            nonlocal obs, info, do_reset, ep_return, ep_length, keys_pressed, l_click, r_click
            obs, info = self.env.reset()
            pygame.event.clear()
            do_reset = False
            ep_return = 0
            ep_length = 0
            keys_pressed = []
            l_click = r_click = False

        obs, info, do_reset, ep_return, ep_length, keys_pressed, l_click, r_click = (None,) * 8

        reset()
        do_wait = False
        should_stop = False

        while not should_stop:
            do_one_step = False
            mouse_x, mouse_y = 0, 0
            pygame.event.pump()

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    should_stop = True

                if event.type == pygame.MOUSEMOTION:
                    mouse_x, mouse_y = event.rel
                    mouse_x *= self.mouse_multiplier
                    mouse_y *= self.mouse_multiplier

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        l_click = True
                    if event.button == 3:
                        r_click = True

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        l_click = False
                    if event.button == 3:
                        r_click = False

                if event.type == pygame.KEYDOWN:
                    keys_pressed.append(event.key)

                elif event.type == pygame.KEYUP and event.key in keys_pressed:
                    keys_pressed.remove(event.key)

                if event.type != pygame.KEYDOWN:
                    continue

                if event.key == pygame.K_RETURN:
                    do_reset = True

                if event.key == pygame.K_PERIOD:
                    do_wait = not do_wait
                    print("Game paused." if do_wait else "Game resumed.")

                if event.key == pygame.K_e:
                    do_one_step = True

                if event.key == pygame.K_m:
                    do_reset = self.env.next_mode()

                if event.key == pygame.K_UP:
                    do_reset = self.env.next_axis_1()

                if event.key == pygame.K_DOWN:
                    do_reset = self.env.prev_axis_1()

                if event.key == pygame.K_RIGHT:
                    do_reset = self.env.next_axis_2()

                if event.key == pygame.K_LEFT:
                    do_reset = self.env.prev_axis_2()

            if do_reset:
                reset()

            if do_wait and not do_one_step:
                continue

            csgo_action = CSGOAction(keys_pressed, mouse_x, mouse_y, l_click, r_click)
            next_obs, rew, end, trunc, info = self.env.step(csgo_action)

            ep_return += rew.item()
            ep_length += 1

            now = time.monotonic()
            dt = now - prev_time
            if dt > 0:
                current_fps = 1.0 / dt
            prev_time = now

            screen.fill(pygame.Color("black"))
            draw_obs(obs)
            draw_action_indicator(csgo_action)

            status_txt = label_font.render(f"{current_fps:.1f} FPS  |  {device_str}", True, pygame.Color(180, 180, 180))
            status_x = x_center - status_txt.get_width() // 2
            status_y = y_center + self.height // 2 + 12
            screen.blit(status_txt, (status_x, status_y))

            if logo_surface is not None:
                screen.blit(logo_surface, (8, 8))

            pygame.display.flip()
            clock.tick(self.fps)

            if end or trunc:
                reset()

            else:
                obs = next_obs

        pygame.quit()
