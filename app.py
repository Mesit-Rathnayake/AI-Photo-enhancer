import gradio as gr

from src.processing.classical_enhancer import process_and_save, process_batch


CUSTOM_CSS = """
.gradio-container {
    max-width: 1350px !important;
    margin: auto;
}

.main-title,
.subtitle {
    text-align: center;
}

.subtitle {
    opacity: 0.82;
    margin-bottom: 20px;
}

/* Premium touches */
button.primary {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
    border: none;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
button.primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4);
}
"""


with gr.Blocks(
    title="AI Photo Enhancer Pro",
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="blue"),
    css=CUSTOM_CSS,
) as demo:

    gr.Markdown(
        "# ✨ AI Photo Enhancer Pro",
        elem_classes=["main-title"],
    )

    gr.Markdown(
        """
        Real-ESRGAN super-resolution with GFPGAN face restoration, 
        mild finishing, and 300-PPI export capabilities.
        """,
        elem_classes=["subtitle"],
    )

    with gr.Tabs():
        with gr.Tab("📸 Single Photo"):
            with gr.Row():
                input_image = gr.Image(
                    label="Original Photo",
                    type="numpy",
                    image_mode="RGB",
                    height=540,
                )

                output_image = gr.Image(
                    label="AI Enhanced Photo",
                    type="numpy",
                    image_mode="RGB",
                    height=540,
                )
            
            enhance_single_btn = gr.Button(
                "✨ Run AI Super-Resolution",
                variant="primary",
                size="lg",
            )
            
            download_single_file = gr.File(
                label="Download AI Enhanced JPEG",
                interactive=False,
            )
            
            single_info = gr.Markdown(
                "Upload a photo and run AI super-resolution."
            )

        with gr.Tab("🗂️ Batch Processing"):
            with gr.Row():
                batch_input = gr.File(
                    file_count="multiple",
                    type="filepath",
                    label="Upload Multiple Photos",
                    height=300,
                )
                
                batch_output = gr.File(
                    label="Download Enhanced Batch (ZIP)",
                    interactive=False,
                    height=300,
                )
                
            enhance_batch_btn = gr.Button(
                "🚀 Process Batch",
                variant="primary",
                size="lg",
            )
            
            batch_info = gr.Markdown(
                "Upload multiple photos. They will be processed and bundled into a ZIP file."
            )

    gr.Markdown("---")
    gr.Markdown("### ⚙️ Processing Settings")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("#### AI Engine Options")

            capture_mode = gr.Dropdown(
                choices=[
                    "Standard",
                    "Wide-angle Distortion Correction",
                    "Digital 2x Quality Recovery",
                ],
                value="Standard",
                label="Camera Correction",
                info=(
                    "Standard skips correction. "
                    "Use Wide-angle Distortion Correction for 24mm phone barrel distortion and edge stretching. "
                    "Use Digital 2x Quality Recovery to clean noise and softness from digital zoom crops."
                ),
            )

            ai_model = gr.Radio(
                choices=[
                    "RealESRGAN x2",
                    "RealESRGAN x4",
                    "RealESRGAN Anime x4",
                    "None",
                ],
                value="RealESRGAN x2",
                label="AI Upscaling Model",
                info=(
                    "Select None to skip AI upscaling and keep original resolution. "
                    "Begin with x2 for balanced detail, or x4 for maximum enlargement."
                ),
            )
            
            face_restoration = gr.Checkbox(
                label="Enable Face Restoration (GFPGAN)",
                value=False,
                info="Fixes blurry or warped faces. Highly recommended for portraits."
            )

            skin_smoothing = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.2,
                step=0.05,
                label="Skin Smoothing",
                info="Softens fine wrinkles inside detected faces. Disabled unless face restoration is enabled.",
            )

            preserve_colors = gr.Checkbox(
                label="Preserve Original Colors",
                value=True,
                info="Keeps the original photo palette while GFPGAN reconstructs facial detail.",
            )

            tile_size = gr.Dropdown(
                choices=[128, 256, 512],
                value=256,
                label="GPU Tile Size",
                info=(
                    "Use 128 if you get a CUDA out-of-memory error. "
                    "Larger tiles can process faster but use more VRAM."
                ),
            )

        with gr.Column():
            gr.Markdown("#### Optional Finishing")

            sharpening_strength = gr.Slider(
                minimum=0.0,
                maximum=0.5,
                value=0.0,
                step=0.05,
                label="Additional Sharpening",
                info="Keep at zero during the first comparison.",
            )

            saturation_adjustment = gr.Slider(
                minimum=-0.3,
                maximum=0.3,
                value=0.0,
                step=0.05,
                label="Saturation Adjustment",
                info="Zero preserves the model output colours.",
            )

            output_ppi = gr.Dropdown(
                choices=[72, 150, 300, 600],
                value=300,
                label="Output PPI",
            )

            jpeg_quality = gr.Slider(
                minimum=70,
                maximum=100,
                value=95,
                step=1,
                label="JPEG Quality",
            )

    # Wire up the buttons
    enhance_single_btn.click(
        fn=process_and_save,
        inputs=[
            input_image,
            ai_model,
            tile_size,
            sharpening_strength,
            saturation_adjustment,
            output_ppi,
            jpeg_quality,
            face_restoration,
            skin_smoothing,
            preserve_colors,
            capture_mode,
        ],
        outputs=[
            output_image,
            download_single_file,
            single_info,
        ],
        show_progress="full",
    )

    enhance_batch_btn.click(
        fn=process_batch,
        inputs=[
            batch_input,
            ai_model,
            tile_size,
            sharpening_strength,
            saturation_adjustment,
            output_ppi,
            jpeg_quality,
            face_restoration,
            skin_smoothing,
            preserve_colors,
        ],
        outputs=[
            batch_output,
            batch_info,
        ],
        show_progress="full",
    )


if __name__ == "__main__":
    demo.queue().launch(
        inbrowser=True,
        show_error=True,
    )