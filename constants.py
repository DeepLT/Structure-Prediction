# constant for sequence preprocessing

res_types = "ACDEFGHIKLMNPQRSTVWY" # residue types

res_to_n = {x: i for i, x in enumerate(res_types)} # create {res : idx} dictionary

# atom_types = ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "NZ", "OH"]

res_to_num = lambda x: res_to_n[x] if x in res_to_n else len(res_to_n) # res string to int