# Fitsmap Creation Rule
rule fmap:
    input:
        'logs/{field}-zfit.log'
    output:
        'logs/{field}-fmap.log'
    log:
        'logs/{field}-fmap.log'
    group:
        lambda wildcards: f'fmap-{groups[wildcards.field]}'
    conda:
        '../envs/fitsmap.yaml'
    # resources:
    #     tasks = lambda wildcards: len(uncal[wildcards.field])
    shell:
        """
        ./workflow/scripts/makeFitsmap.py {wildcards.field} --slowsegmap --ncpu {resources.cpus_per_task} > {log} 2>&1
        cp FIELDS/{wildcards.field}/fitsmap/RGB.png FIELDS/{wildcards.field}/fitsmap/map/
        tar -cf FIELDS/{wildcards.field}/fitsmap/{wildcards.field}.tar -C FIELDS/{wildcards.field}/fitsmap/ {wildcards.field}/
        """