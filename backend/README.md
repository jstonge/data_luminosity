# Data luminosity

This project builds a model to classify data availability statements (DAS) in papers. 

Our data comes from the `s2orc` dataset on our `MongoDB` (see readme below)

We define DAS as any statement regarding the author's _intention_ to share their data, going beyond simple keyword search.

Our definition is made explicit by our hand-annotated data. 

We use `label-studio` to manage the data labeling process, accelerated with LLM preannotations. 

<details><summary>s2orc README</summary>
Semantic Scholar Academic Graph Datasets

The "s2orc" dataset contains parsed full-body text from selected papers.

A subset of this data was previously released (in a different format) as S2ORC https://github.com/allenai/s2orc

The body text is parsed from PDF documents using Grobid, documented at https://grobid.readthedocs.io.
Its output is converted from XML into a single string with a set of annotation spans.

SCHEMA
 - externalIds: IDs of this paper in different catalogs
 - content:
   - source:
           - pdfUrls: URLs to the PDF
           - oaInfo: license/url/status information from Unpaywall
   - text: Full body text as a single string
   - annotations: Annotated spans of the full body text


LICENSE
This collection is licensed under ODC-BY. (https://opendatacommons.org/licenses/by/1.0/)

By downloading this data you acknowledge that you have read and agreed to all the terms in this license.

ATTRIBUTION
When using this data in a product or service, or including data in a redistribution, please cite the following paper:

@inproceedings{lo-wang-2020-s2orc,
    title = "{S}2{ORC}: The Semantic Scholar Open Research Corpus",
    author = "Lo, Kyle  and Wang, Lucy Lu  and Neumann, Mark  and Kinney, Rodney  and Weld, Daniel",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.447",
    doi = "10.18653/v1/2020.acl-main.447",
    pages = "4969--4983"
}

</details>

