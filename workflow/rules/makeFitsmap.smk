# Fitsmap Creation Rule
rule fmap:
    input:
        'FIELDS/{field}/logs/zfit.out'
    output:
        'FIELDS/{field}/logs/fmap.out'
    conda:'../envs/fitsmap.yaml'
    log:'logs/Fitsmap_{field}.log'
    shell:
        """
        ./workflow/scripts/makeFitsmap.py {wildcards.field} --slowsegmap --ncpu $SLURM_CPUS_PER_TASK
        """