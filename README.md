# Code_for_Spectral_Clustering_Methods_for_Signed_Hypergraphs
This is the code and dataset used in the simulation experiments of the article "Spectral Clustering Methods for Signed Hypergraphs".
Since the amount of experimental code is relatively large, to avoid confusion, three branches have been created to execute different experimental tasks respectively. In addition, the dataset used in the paper is also provided in the branch "dataset".

# 2-way_partition
This part is used for the 2-partition experiment. Among them, the scripts starting with "main" are the main scripts, those starting with "statistic" are supplementary experiments for statistical significance analysis, and those starting with "house" are for preprocessing and visualization of the House dataset. Other files are auxiliary methods classified according to functional modules, which are convenient for the main program to call.

# hierachical_partition_gene
This part is the experimental code for hierarchical clustering on the generated dataset. Among them, "hierachical_iter" and "hierachical_main" are the main scripts, and the others are newly written (or borrowed from the 2-partitioning) auxiliary methods, which are convenient for the main program to call.

# hierachical_partition_real
This part is similar to the "hierarchical_partition_gene" branch, except that it is used for hierarchical clustering experiments on real datasets.

# dataset
This section provides six real dataset files used in the text. Among them, for four citation networks, due to the lack of associated positive and negative information, they need to be generated manually. The codes for the relevant methods are included in each branch.

# Contact information
Since the code and dataset have only been roughly organized, if you encounter any problems during use, please feel free to contact us at the following email address: lucky0128mw@163.com
