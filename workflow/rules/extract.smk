# Extraction Rule
rule extract:
    input:
        'FIELDS/{field}/logs/contam.out'
    output:
        'FIELDS/{field}/logs/extr.out'
    conda:'../envs/grizli.yaml'
    log:'logs/Extract_{field}.log'
    shell:
        """
        ./workflow/scripts/extract.py {wildcards.field} --ncpu {threads}
        """