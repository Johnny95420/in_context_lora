check_instruction = """Please help me verify whether the following images meet the required format: 
- The first image contains only clothing, accessories, or shoes, with no person present.
- The second image contains only clothing, accessories, or shoes, with no person present.
- The third image contains a person.
If there is no issue, return true. If there is an issue, return false. No other content is needed."""

instruction = """
Task:
You are a professional fashion captioner working on AI training data for virtual try-on systems.
**Words limit is very importtant**

You are given a stitched image consisting of 3 fashion-related photos:

- <IMAGE 1>: A standalone image of a fashion item (e.g., a T-shirt, dress, jacket) showing one view.
- <IMAGE 2>: Another view of the **same item**, from a different angle or distance.
- <IMAGE 3>: A human model wearing the same item in a styled setting.

Your task is to generate a **single paragraph**, written in fluent, descriptive language suitable for a high-end fashion catalog or AI captioning dataset. Your output must follow the structure below and stay **within 600 words**.

---
1. Begin with a **one-sentence summary** of the clothing item and its design (**Constrain 100 words**):
   - What type of garment it is (e.g., dress, coat),
   - Its color, material, silhouette,
   - General aesthetic or usage context (e.g., summer casual, urban chic).
---

2. Then write three image-specific descriptions:

<IMAGE 1> Describe the garment shown in this image. Focus on the visible design, color, fabric, texture, silhouette, and structural details. Indicate the visible part (e.g., front view, close-up).
- Always refer to the garment consistently as **<Brand> <ItemType>**, such as <Brand> dress, <Brand> blouse, <Brand> shoes> — this special token should **not** be replaced with a real brand.
- **Constrain 100 words**

<IMAGE 2> Describe the same <Brand> <ItemType> from a second view, emphasizing design features not visible in <IMAGE 1> — for example: back detail, sleeve construction, fabric movement, buttons, or fastenings.
- **Constrain 100 words**

<IMAGE 3> Describe the model in high detail:
- Apparent **gender**, **race/ethnicity**, **hair color and style**, **body shape**, and **pose**.
- Describe how the <Brand> <ItemType> fits the model and how it interacts with their body (e.g., hugs waist, flows from shoulders).
- Mention **any other visible fashion items** worn: additional clothing, accessories, shoes, jewelry, etc.
- Describe the **background, lighting**, and overall visual tone or mood (e.g., natural sunlight, studio flash, minimal backdrop).
- **Constrain 100 words**

---
Response Constrain:
- Remember to add <IMAGE number> before describing each image.
- Keep all references to the garment consistent using the <Brand> <ItemType> format — do not change it or replace it with brand names.
- Keep the full paragraph **under 400 words**. After generating the translation, carefully review it and make sure the total word count does not exceed 400 words. If it exceeds the limit, regenerate and verify the result. Do not output the checking results in your response.

Now generate the caption for the stitched image accordingly.
"""
