# Fitsmap Creation Rule
rule fmap:
    input:
        'logs/{field}-zfit.log'
    output:
        'logs/{field}-fmap.log'
    conda:
        '../envs/fitsmap.yaml'
    log:
        'logs/{field}-fmap.log',
    shell:
        """
        ./workflow/scripts/makeFitsmap.py {wildcards.field} --slowsegmap --ncpu {threads} &> {log}
        """