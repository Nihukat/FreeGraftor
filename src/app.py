import os
os.environ['FLUX_DEV'] = '/workspace/yzb/pretrained/huggingface/black-forest-labs/FLUX.1-dev'
os.environ['GROUNDING_DINO'] = '/workspace/yzb/pretrained/huggingface/IDEA-Research/grounding-dino-tiny'
os.environ['SAM'] = '/workspace/yzb/pretrained/huggingface/HCMUE-Research/SAM-vit-h/sam_vit_h_4b8939.pth'


import gradio as gr
from collections import defaultdict
from pipeline import FreeGraftorPipeline, ConceptConfig, GenerationConfig

MAX_REFERENCES = 5

pipeline = FreeGraftorPipeline()

with gr.Blocks(title="FreeGraftor Image Generation") as demo:
    gr.Markdown("# FreeGraftor: Training-Free Cross-Image Feature Grafting for Subject-Driven Text-to-Image Generation")
    
    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(label="Prompt", placeholder="Enter your description...")
            
            with gr.Group():
                gr.Markdown("## Reference Subjects")
                
                reference_sections = []
                for i in range(MAX_REFERENCES):
                    with gr.Row(visible=i == 0) as row:
                        img = gr.Image(
                            label=f"Reference {i+1}", 
                            type="filepath", 
                            interactive=True
                        )
                        
                        with gr.Column(scale=2, visible=False) as params_col:
                            class_name = gr.Textbox(label="Class Name", placeholder='Enter subject class (e.g., "cat")')
                            with gr.Accordion("Affine Transformation", open=True):
                                scale = gr.Slider(label="Scale", minimum=0.1, maximum=5.0, value=1.0, step=0.1)
                                x_bias = gr.Slider(label="X Bias", minimum=-1024, maximum=1024, value=0, step=1)
                                y_bias = gr.Slider(label="Y Bias", minimum=-1024, maximum=1024, value=0, step=1)
                                angle = gr.Slider(label="Rotation Angle", minimum=0, maximum=360, value=0, step=1)
                                flip = gr.Checkbox(label="Horizontal Flip", value=False)
                                alignment = gr.Radio(label="Alignment", choices=["center", "top", "bottom", "left", "right", "top-left", "top-right", "bottom-left", "bottom-right"], value="center")
                        
                        reference_sections.append({
                            "row": row,
                            "image": img,
                            "params_col": params_col,
                            "class_name": class_name,
                            "scale": scale,
                            "x_bias": x_bias,
                            "y_bias": y_bias,
                            "angle": angle,
                            "flip": flip,
                            "alignment": alignment
                        })

            with gr.Accordion("Advanced Settings", open=True):
                with gr.Row():
                    guidance = gr.Slider(label="CFG Scale", minimum=1.0, maximum=10.0, value=3.0, step=0.5)
                    num_steps = gr.Slider(label="Inference Steps", minimum=1, maximum=50, value=25, step=5)
                
                with gr.Row():
                    start_seed = gr.Number(label="Start Seed", value=0)
                    num_images = gr.Slider(label="Number of Images", minimum=1, maximum=10, value=1, step=1)
                
                with gr.Row():
                    width = gr.Slider(label="Width", minimum=512, maximum=2048, value=1024, step=64)
                    height = gr.Slider(label="Height", minimum=512, maximum=2048, value=1024, step=64)
                    
                with gr.Row():
                    start_inject_step = gr.Slider(label="Start Inject Step", minimum=0, maximum=50, value=0, step=1)
                    end_inject_step = gr.Slider(label="End Inject Step", minimum=0, maximum=50, value=25, step=1)
                
                with gr.Row():
                    start_inject_block = gr.Slider(label="Start Inject Block", minimum=0, maximum=56, value=0, step=1)
                    end_inject_block = gr.Slider(label="End Inject Block", minimum=0, maximum=56, value=56, step=1)
                
                with gr.Row():
                    sim_threshold = gr.Slider(label="Similarity Threshold", minimum=0.0, maximum=1.0, value=0.2, step=0.1)
                    cyc_threshold = gr.Slider(label="Cycle Consistency Threshold", minimum=0.0, maximum=5.0, value=1.5, step=0.1)
                
                inject_match_dropout = gr.Slider(label="Match Dropout", minimum=0.0, maximum=1.0, value=0.2, step=0.1)
            
                with gr.Row():
                    offload = gr.Checkbox(label="Offload", value=False)
                    clear_image_cache = gr.Checkbox(label="Clear Image Cache", value=False)
                    clear_image_info_cache = gr.Checkbox(label="Clear Image Info Cache", value=False)
            
            generate_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=3):
            gallery = gr.Gallery(label="Generated Images", columns=3, object_fit="contain", height=800)
            gr.Markdown("Generation Settings Summary:")
            settings_summary = gr.JSON(label="Current Configuration")

    for i, section in enumerate(reference_sections):
        def update_ui_on_image_change(img, i=i):
            updates = []
            
            updates.append(gr.update(visible=img is not None))
            
            if i < MAX_REFERENCES - 1:
                updates.append(gr.update(visible=img is not None))
            else:
                updates.append(gr.update(visible=False))
                
            return updates
        
        outputs = [section["params_col"]]
        
        if i < MAX_REFERENCES - 1:
            outputs.append(reference_sections[i+1]["row"])
        else:
            outputs.append(section["params_col"])
            
        section["image"].change(
            fn=update_ui_on_image_change,
            inputs=[section["image"]],
            outputs=outputs
        )
    
    def generate_images(progress=gr.Progress(), prompt="", num_images=1, guidance=3.0, num_steps=25, 
                      start_seed=0, width=1024, height=1024, 
                      start_inject_step=0, end_inject_step=25, start_inject_block=0, end_inject_block=56, 
                      sim_threshold=0.2, cyc_threshold=1.5, inject_match_dropout=0.2, 
            offload=False, clear_image_cache=False, clear_image_info_cache=False, *concept_params):
        images = concept_params[:MAX_REFERENCES]
        other_params = concept_params[MAX_REFERENCES:]
        
        valid_images = [img for img in images if img is not None]
        valid_count = len(valid_images)
        
        config_summary = {
            "prompt": prompt,
            "num_images": num_images,
            "guidance_scale": guidance,
            "steps": num_steps,
            "seed": start_seed,
            "resolution": f"{width}x{height}",
            "subjects": []
        }
        
        inject_block_ids = list(range(start_inject_block, end_inject_block + 1))
        
        info = {
            'guidance': guidance,
            'num_steps': num_steps,
            'start_inject_step': start_inject_step,
            'end_inject_step': end_inject_step,
            'inject_block_ids': inject_block_ids,
            'sim_threshold': sim_threshold,
            'cyc_threshold': cyc_threshold,
            'inject_match_dropout': inject_match_dropout,
            'width': width,
            'height': height
        }
        
        concept_configs = []
        
        for i in range(valid_count):
            param_start = i * 7
            if valid_images[i] is not None:
                config_dict = {
                    "class_name": other_params[param_start],
                    "image_path": valid_images[i],
                    "scale": other_params[param_start+1],
                    "x_bias": other_params[param_start+2],
                    "y_bias": other_params[param_start+3],
                    "angle": other_params[param_start+4],
                    "flip": other_params[param_start+5],
                    "alignment": other_params[param_start+6],
                }
                concept_configs.append(ConceptConfig(**config_dict))
                
                config_summary["subjects"].append({
                    "class_name": config_dict["class_name"],
                    "scale": config_dict["scale"],
                    "affine_transformation": {
                        "x": config_dict["x_bias"],
                        "y": config_dict["y_bias"], 
                        "angle": config_dict["angle"],
                        "flip": config_dict["flip"],
                        "alignment": config_dict["alignment"]
                    }
                })

        outputs = []
        
        def progress_callback(step, total_steps, image_index, total_images):
            progress_pct = step / total_steps
            overall_pct = (image_index + progress_pct) / total_images
            progress(overall_pct, desc=f"Generating image {image_index+1}/{total_images}: Step {step}/{total_steps}")
            html = f"""
            <div style='text-align: center; margin: 10px 0'>
                <div style='font-size: 14px; margin-bottom: 5px;'>
                    Generating image {image_index+1} of {total_images}
                </div>
                <div style='width: 100%; background-color: #f3f3f3; height: 20px; border-radius: 10px; overflow: hidden;'>
                    <div style='height: 100%; width: {progress_pct*100}%; background-color: #4CAF50; border-radius: 10px;'></div>
                </div>
                <div style='font-size: 12px; margin-top: 5px;'>
                    Step {step}/{total_steps} (Overall progress: {overall_pct*100:.1f}%)
                </div>
            </div>
            """
            return html
        
        current_progress_html = None
        for i in range(int(num_images)):
            seed = int(start_seed) + i
            info["seed"] = seed
            
            current_progress_html = progress_callback(0, num_steps, i, num_images)
            yield [], config_summary
            pipeline.requires_offload = offload
            result = pipeline(
                prompt=prompt,
                concept_configs=concept_configs,
                clear_image_cache=clear_image_cache,
                clear_image_info_cache=clear_image_info_cache,
                config=GenerationConfig(**info),
                callback=lambda step, total_steps: progress_callback(step, total_steps, i, num_images)
            )
            outputs.append(result)
            
            current_progress_html = progress_callback(num_steps, num_steps, i, num_images)
            yield outputs, config_summary
        
        yield outputs, config_summary

    inputs = [
        prompt,
        num_images,
        guidance,
        num_steps,
        start_seed,
        width,
        height,
        start_inject_step,
        end_inject_step,
        start_inject_block,
        end_inject_block,
        sim_threshold,
        cyc_threshold,
        inject_match_dropout,
        offload,
        clear_image_cache,
        clear_image_info_cache,
    ]
    
    for sec in reference_sections:
        inputs.append(sec["image"])
    
    for sec in reference_sections:
        inputs.extend([
            sec["class_name"],
            sec["scale"],
            sec["x_bias"],
            sec["y_bias"],
            sec["angle"],
            sec["flip"],
            sec["alignment"]
        ])
    
    generate_btn.click(
        fn=generate_images,
        inputs=inputs,
        outputs=[gallery, settings_summary]
    )

if __name__ == "__main__":
    demo.launch(share=True)