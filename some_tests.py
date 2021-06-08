from fkg_mini_project import FKGMiniProject

ontology = "carcinogenesis.owl"
lps = "kg-mini-project-train.ttl"
project = FKGMiniProject(ontology, lps)

# print some information about the learning problems and the ontology
project.print_infos_ontology()
project.print_infos_lps()

# test printing of the result file on the first learning problem
lp = project.lps[0]
lp_name = lp['lp']
pos = lp['pos']
neg = lp['neg']
project.write_result_file(lp_name, pos, neg)