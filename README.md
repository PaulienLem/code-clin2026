#  Unsupervised Clustering Techniques for Historical Greek

This is the repository accompanying 
the paper *From Orthography to Semantics: Large-Scale Unsupervised Textual Similarity in Historical Greek*, 
which was submitted for the journal 
[Computational Linguistics in the Netherlands (CLIN)](https://www.clinjournal.org).
Due to copyright reasons, the full dataset cannot be deployed here, 
but a small subset of the [Database of Byzantine Book Epigrams](https://www.dbbe.ugent.be) has been included to allow running the DBBE sections of the notebook. Replacing this file with a different csv following the same structure, should allow running the script using other data.

-------
 ## Running the scripts

- Use [virtualenv](https://virtualenv.pypa.io/en/latest/user_guide.html) to generate a virtual **python3.11** environment
- Install the required dependencies using `pip install -r requirements.txt`
- Run the "DBBE" section of `clin_demo_similarity_clustering.ipynb`. These sections use the small demo dataset included in this repository. 
