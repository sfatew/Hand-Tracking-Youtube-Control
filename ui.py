import dearpygui.dearpygui as dpg
import cv2 as cv
import numpy as np
from utils.ui_helpers import *
import threading

dpg.create_context()
dpg.create_viewport(title='Hand tracking computer control', width=600, height=500)
dpg.setup_dearpygui()

vid = cv.VideoCapture(0)
ret, frame = vid.read()

if not ret:
    print("Error: Could not read from camera.")
    vid.release()
    dpg.destroy_context()
    exit()

frame_width, frame_height, texture_data = create_texture_data(frame)  # create texture data from the frame

print(f"Frame width: {frame_width}, Frame height: {frame_height}, Webcam FPS: {vid.get(cv.CAP_PROP_FPS)}")
print(f"texture_data shape: {texture_data.shape}, dtype: {texture_data.dtype}")

state = ProgramState(fps=vid.get(cv.CAP_PROP_FPS), skip_interval=1)

with dpg.texture_registry(show=True):
    dpg.add_raw_texture(frame_width, frame_height, texture_data, tag="texture_tag", format=dpg.mvFormat_Float_rgb)

with dpg.window(label="Main Window", width=800, height=450, tag="Primary Window"):
    dpg.add_text("FPS: is not updated", tag="fps_text")
    dpg.add_slider_float(label="time to collect frames", 
                         default_value=state.time_to_collect_frames, 
                         max_value=3.0, 
                         min_value=0.5, 
                         width=300,
                         tag="time_to_collect_frames_slider")
    dpg.add_slider_float(label="cooldown time",
                         default_value=state.cooldown_time, 
                         max_value=5.0, 
                         min_value=0.5, 
                         width=300,
                         tag="cooldown_time_slider")
    dpg.add_button(label="Set values", callback=state.set_values_from_sliders)
    dpg.add_radio_button(label="Select Gesture Model", items=model_list, horizontal=True, default_value=model_list[0], tag="model_selector", callback=state.set_model)
    dpg.add_checkbox(label="Image Enhancement enabled", default_value=state.image_enhancement_enabled, tag="image_enhancement_checkbox", callback=state.toggle_image_enhancement)
    dpg.add_text(state.status_text, tag="status_text", color=(255, 0, 0))
    dpg.add_text(f"Last gesture: {state.pred_label}", tag="last_gesture_text", color=(0, 255, 0))
    with dpg.group(horizontal=True):
        dpg.add_image("texture_tag")
        with dpg.plot(label="Color Histogram", height=-1, width=-1):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label="Pixel Value")
            y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Frequency")

            for col in ('b', 'g', 'r'):
                dpg.add_line_series(
                    list(range(256)),
                    hist_data[col],
                    label=col.upper(),
                    tag=f"{col}_series",
                    parent=y_axis
                )

dpg.show_metrics()
dpg.show_viewport()


# Start a thread to update FPS
#threading.Thread(target=state.update_fps_continuously, args=(vid,), daemon=True).start()

state.update_fps(vid)  # initial FPS update

while dpg.is_dearpygui_running(): # collect -> predict -> cooldown -> reset buffer -> collect

    ret, frame = vid.read()

    if not ret:
        print("Error: Could not read from camera.")
        dpg.stop_dearpygui()
        break

    if state.image_enhancement_enabled:
        frame = enhance_image(frame)
    

    if state.is_ready_for_prediction():
            state.is_predicting = True
            threading.Thread(target=state.run_inference, daemon=True).start()
    else:
        state.add_frame(frame)

    if state.is_predicting:
        state.change_status_text("Model is predicting...")

    if not state.is_predicting and state.last_prediction_time is not None:
        remaining = state.cooldown_time - (time.time() - state.last_prediction_time)
        if remaining > 0:
            state.change_status_text(f"Cooldown: {remaining:.1f}s")
            state.frame_buffer.clear()
            state.frame_counter = 0

    _, _, texture_data = create_texture_data(frame)  # create texture data from the frame

    # create histogram data for the current frame
    update_histogram(frame)

    dpg.set_value("texture_tag", texture_data)  # update the texture with the new frame data
    dpg.render_dearpygui_frame()

vid.release()
dpg.destroy_context()