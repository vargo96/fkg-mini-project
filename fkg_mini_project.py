# Reads the TTL file and returns the positive and negative instances
# for all the learning problems as list of dictionaries
def read_and_parse_ttl_file(file_path):
    lp_instance_list = []
    with open(file_path, "r") as lp_file:
        for line in lp_file:
            if line.startswith("lpres:"):
                lp_key = line.split()[0].split(":")[1]
            elif line.strip().startswith("lpprop:excludesResource"):
                exclude_resource_list = line.strip()[23:].split(",")
                exclude_resource_list = [individual.replace(";", "").replace("carcinogenesis:", "").strip()
                                         for individual in exclude_resource_list]
            elif line.strip().startswith("lpprop:includesResource"):
                include_resource_list = line.strip()[23:].split(",")
                include_resource_list = [individual.replace(".", "").replace("carcinogenesis:", "").strip()
                                         for individual in include_resource_list]
                lp_instance_list.append({"lp": lp_key, "pos": include_resource_list, "neg": exclude_resource_list})

    return lp_instance_list


print(read_and_parse_ttl_file('kg-mini-project-train.ttl'))
