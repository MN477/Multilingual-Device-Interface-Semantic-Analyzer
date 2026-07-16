from paddleocr import DocPreprocessor

# 1. This class only handles unwarping and rotation
preprocessor = DocPreprocessor(
    use_doc_orientation_classify=True,
    use_doc_unwarping=True,
    doc_orientation_classify_model_name="PP-LCNet_x1_0_doc_ori" # Default orientation model
)

# 2. Run the unwarping and rotating
image_path = "input_images/LCD1.jpg"
output = preprocessor.predict(image_path)

# 3. Save the beautiful, flattened, upright image
for res in output:
    res.save_to_img(save_path="./corrected_output/")