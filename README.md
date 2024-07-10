# Text based Image Search & Retrieval
<ul>
  <li>Implemented OpenAI Multi Modal CLIP Architecture from research paper using pretrained image encoder Resnet50 and text encoder all-mini-LM-V12 to generate image and sentence embeddings respectively
</li>
  <li> Trained embedding extractor heads with CLIP Architecture on MS COCO 2017 to generate multi modal embedding extractors using Cross Entropy Loss with Adam Optimizer & Cosine Annealing
</li>
  <li> Generated image embeddings & used ChromaDB collections to index image embeddings for faster text based image retrieval
</li>
</ul>
