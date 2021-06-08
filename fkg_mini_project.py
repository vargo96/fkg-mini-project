from owlready2 import World, Thing
from rdflib import Namespace, Graph, Literal, URIRef


class FKGMiniProject:

    def __init__(self, ontology_path, lps_path):
        self.ontology_path = ontology_path
        self.lps_path = lps_path
        self.onto = World().get_ontology(self.ontology_path).load()
        self.lps = self._read_and_parse_ttl_file(self.lps_path)


    def _read_and_parse_ttl_file(self, lps_path):
        lp_instance_list = []
        with open(lps_path, "r") as lp_file:
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


    def fit(self, lp):
        pass


    def score(self, lp):
        pass


    def write_result_file(self, lp_name, pos, neg, result_file="result.ttl"):
        NS_CAR = Namespace("http://dl-learner.org/carcinogenesis#")
        NS_RES = Namespace("https://lpbenchgen.org/resource/")
        NS_PROP = Namespace("https://lpbenchgen.org/property/")

        g = Graph()
        g.bind('carcinogenesis', NS_CAR)
        g.bind('lpres', NS_RES)
        g.bind('lpprop', NS_PROP)

        g.add((NS_RES.result_1pos, NS_PROP.belongsToLP, Literal(True)))
        g.add((NS_RES.result_1pos, NS_PROP.pertainsTo, URIRef(NS_RES + lp_name)))
        for p in pos:
            g.add((NS_RES.result_1pos, NS_PROP.resource, URIRef(NS_CAR + p)))

        g.add((NS_RES.result_1neg, NS_PROP.belongsToLP, Literal(False)))
        g.add((NS_RES.result_1neg, NS_PROP.pertainsTo, URIRef(NS_RES + lp_name)))
        for n in neg:
            g.add((NS_RES.result_1neg, NS_PROP.resource, URIRef(NS_CAR + n)))

        g.serialize(destination=lp_name + "_"+ result_file, format='turtle')


    def print_infos_ontology(self):
        print("#"*50)
        print(f"Number classes: {len(list(self.onto.classes()))}")
        instances = set()
        for c in self.onto.classes():
            print(f"\t {c.name}")
            instances.update(c.instances(world=self.onto.world))
        print(f"Number object properties: {len(list(self.onto.object_properties()))}")
        for p in self.onto.object_properties():
            print(f"\t {p.name}")
        print(f"Number data properties: {len(list(self.onto.data_properties()))}")
        for p in self.onto.data_properties():
            print(f"\t {p.name}")
        print(f"Number individuals: {len(instances)}")
        print("#"*50)


    def print_infos_lps(self):
        print("#"*50)
        print(f"Number LPs: {len(self.lps)}")
        for lp in self.lps:
            print(f"LP ({lp['lp']}): PositiveEX - {len(lp['pos'])} | NegativeEX - {len(lp['neg'])}")
        print("#"*50)