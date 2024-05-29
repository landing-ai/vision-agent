# Generate Masks for DINOv

This application allows you to generate masks to use for the DINOv tool. To get started
install the requirements by running:

```bash
pip install -r requirements.txt
```

Then you can run the streamlit app by running:

```bash
streamlit run app.py
```

From here you can upload an image, paint a mask over the image, and then save the mask.
This can be used as input for the DINOv tool.

```python
import vision_agent as va

data = {
    "prompt": [{"mask": "baggage.png", "image": "baggage_mask.png"}],
    "image": "baggage2.png",
}
tool = va.tools.easytool_tools.DINOv()
output = res(**data)
image = va.utils.image_utils.overlay_masks("baggage2.png", output)
image = va.utils.image_utils.overlay_bboxes(image, output)
image.show()
```
