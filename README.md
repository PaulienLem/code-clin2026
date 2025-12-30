#  Unsupervised Clustering Techniques for Historical Greek

This is the repository accompanying 
the paper *From Orthography to Semantics: Large-Scale Unsupervised Textual Similarity in Historical Greek*, 
which was submitted for the journal 
[Computational Linguistics in the Netherlands (CLIN)](https://www.clinjournal.org).
Due to copyright reasons, the full dataset cannot be deployed here, 
but a small subset of the [Database of Byzantine Book Epigrams](https://www.dbbe.ugent.be) has been included to allow running the DBBE sections of both  `orthographic_similarity.ipynb` and `semantic_similarity.ipynb`. Replacing this file with a different csv following the same structure, should allow running both scripts against other data.

-------
 ## Running the scripts

- Use [virtualenv](https://virtualenv.pypa.io/en/latest/user_guide.html) to generate a virtual **python3.11** environment
- Install the required dependencies using `pip install -r requirements.txt`
- Run the "DBBE" section of the `orthographic_similarity.ipynb` script or the `semantic_similarity.ipynb` script. These sections use the small demo dataset included in this repository. Note that the semantic similarity script requires a GPU environment. The orthographic similarity notebook can be run on a regular personal computer without GPU. 
