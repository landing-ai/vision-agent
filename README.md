# Large Multimodal Model Tools
LMM-Tools (Large Multmodal Model Tools) is a minimal library that helps you utilize multimodal models to organize and structure your image data. One of the problems of dealing with image data is it can be difficult to organize and quickly search. For example, you might have a bunch of pictures of houses and want to count how many yellow houses you have, or how many houses with adobe roofs. This library utilizes LMMs to help create these tags or descriptions and allow you to search over them, or use them in a database to do other operations.

## Getting Started
### LMMs
To get started you can create an LMM and start generating text from images. The following code will grab the LLaVA-1.6 34B model and generate a description of the image you pass it.

```python
import lmm_tools as lmt

model = lmt.lmm.get_model("llava")
model.generate("Describe this image", "image.png")
>>> "A yellow house with a green lawn."
```

**WARNING** We are hosting the LLaVA-1.6 34B model, if it times out please wait ~5-10 min for the server to warm up as it shuts down when usage is low.

### DataStore
You can use the `DataStore` class to store your images, add new metadata to them such as descriptions, and search over different columns.

```python
import lmm_tools as lmt
import pandas as pd

df = pd.DataFrame({"image_paths": ["image1.png", "image2.png", "image3.png"]})
ds = lmt.data.DataStore(df)
ds = ds.add_lmm(lmt.lmm.get_model("llava"))
ds = ds.add_embedder(lmt.emb.get_embedder("sentence-transformer"))

ds = ds.add_column("descriptions", "Describe this image.")
```

This will use the prompt you passed, "Describe this image.", and the LMM to create a new column of descriptions for your image. Your data will now contain a new column with the descriptions of each image:

| image\_paths | image\_id | descriptions |
| --- | --- | --- |
| image1.png | 1 | "A yellow house with a green lawn." |
| image2.png | 2 | "A white house with a two door garage." |
| image3.png | 3 | "A wooden house in the middle of the forest." |

You can now create an index on the descriptions column and search over it to find images that match your query.

```python
ds = ds.build_index("descriptions")
ds.search("A yellow house.", top_k=1)
>>> [{'image_paths': 'image1.png', 'image_id': 1, 'descriptions': 'A yellow house with a green lawn.'}]
```

You can also create other columns for you data such as `is_yellow`:

```python
ds = ds.add_column("is_yellow", "Is the house in this image yellow? Please answer yes or no.")
```

which would give you a dataset similar to this:

| image\_paths | image\_id | descriptions | is\_yellow |
| --- | --- | --- | --- |
| image1.png | 1 | "A yellow house with a green lawn." | "yes" |
| image2.png | 2 | "A white house with a two door garage." | "no" |
| image3.png | 3 | "A wooden house in the middle of the forest." | "no" |
