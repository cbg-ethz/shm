import scipy as sp
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def _load_data(infile, normalize):
    dat = pd.read_csv(infile, sep="\t")
    if normalize:
        print("Taking log-fold change")
        dat["cm"] = sp.mean(dat[["c1", "c2"]].values, axis=1)
        dat["r1"] = sp.log(dat["r1"].values / dat["cm"].values)
        dat["r2"] = sp.log(dat["r2"].values / dat["cm"].values)
    dat = (dat[["Condition", "Gene", "sgRNA", "r1", "r2"]]
           .query("Gene != 'Control'")
           .melt(id_vars=["Gene", "Condition", "sgRNA"],
                 value_vars=["r1", "r2"],
                 var_name="replicate",
                 value_name="counts")
           .sort_values(["Gene", "Condition", "sgRNA", "replicate"]))
    dat["sgRNA"] = LabelEncoder().fit_transform(dat["sgRNA"].values)
    if not normalize:
        dat["counts"] = sp.floor(dat["counts"].values)
    return dat


@click.command()
@click.argument("infile", type=str)
@click.argument("outfile", type=str)
@click.option('--normalize', '-n', is_flag=True)
@click.option('--keep-burnin', '-k', is_flag=True)
@click.option('--filter', '-f', is_flag=True)
@click.option("--model-type", type=click.Choice(models.keys()), default="shm")
@click.option("--ntune", type=int, default=50)
@click.option("--nsample", type=int, default=100)
@click.option("--ninit", type=int, default=1000)
def run(infile, outfile, normalize, keep_burnin, filter,
        model_type, ntune, nsample, ninit):
    read_counts = _load_data(infile, normalize)
    if filter:
        print("Filtering by genes")
        read_counts = read_counts.query("Gene == 'BCR' | Gene == 'PSMB1'")
    model, genes, gene_conds = models[model_type](read_counts, normalize)

    if not keep_burnin:
        print("Removing burning")
    with model:
        trace = pm.sample(nsample, tune=ntune, init="advi", n_init=ninit,
                          chains=4, random_seed=42,
                          discard_tuned_samples=not keep_burnin)

    pm.save_trace(trace, outfile + "_trace", overwrite=True)
    _plot(model, trace, outfile, genes, gene_conds, ntune, nsample,
          model_type, keep_burnin, read_counts)


if __name__ == "__main__":
    run()
