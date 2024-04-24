### LMMs
One of the problems of dealing with image data is it can be difficult to organize and
search. For example, you might have a bunch of pictures of houses and want to count how
many yellow houses you have, or how many houses with adobe roofs. The vision agent
library uses LMMs to help create tags or descriptions of images to allow you to search
over them, or use them in a database to carry out other operations.

To get started, you can use an LMM to start generating text from images. The following
code will use the LLaVA-1.6 34B model to generate a description of the image you pass it.

```python
import vision_agent as va

model = va.lmm.get_lmm("llava")
model.generate("Describe this image", "image.png")
>>> "A yellow house with a green lawn."
```

**WARNING** We are hosting the LLaVA-1.6 34B model, if it times out please wait ~3-5
min for the server to warm up as it shuts down when usage is low.
