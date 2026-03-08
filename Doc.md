//26-02-2026
Installed Matplotlib and visualized the graph we created.

//27-02-2026
Expanded the database, now the pages from database will not just show citation with pages on database only, their citations with external pages will also be shown.
Created a real graph of the papers.

//28-02-2026
Verified the correctness of graph by making a degree distribution graph and verified it using power law.
Created a Comparision method using "PaperRank", in which papers will be given rank based on the number of citations of the paper.

//01/03/2026
Cleaned the database, and added meaningful data.
Added GPU processsing for the SciBert Embeddings.

//03/03/2026
Did some changes in build graph,to show better analytics.
Also added vector emeddings to each paper.

//04/03/26
Created a proper pipeline for retrieval of top relevant papers and also for retireval changed the criteria from just citation to both citation count and pagerank

//05/03/26
Implemented hybrid research paper retrieval by combining SciBERT semantic similarity (FAISS vector search) with citation-based PageRank from the citation graph.
This improves search quality by ranking papers not only by textual relevance but also by their influence in the research network.
(ISSE YE HELP HOTI HAI KI PAPER SIRF TOPIC SE RELEVANT HI NHI, BALKI USKE ANDAR KI QUALITY CHECK BHI HOTI HAI using a formula of Hybrid = alpha x semantic + beta x pagerank, where alpha = 0.7 and beta = 0.3,)

//06/03/26
Converting PostgreSQL citations to PyTorch graph.(src/gnn/build_graph_dataset)
We generated graph-aware embeddings for each paper using a Graph Convolutional Network (GCN) trained on the citation graph. These embeddings capture structural relationships between papers, enabling the retrieval system to consider citation context in addition to semantic similarity and PageRank.

//07/03/26
Did some changes in query_faiss and mainly worked on Evaluation metrics.




