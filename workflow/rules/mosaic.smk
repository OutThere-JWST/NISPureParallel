# Mosaic Rule
rule mosaic:
    input:
        'FIELDS/{field}/logs/proc.out'
    output:
        'FIELDS/{field}/logs/mos.out'
    conda:'../envs/grizli.yaml'
    log:'logs/Mosaic_{field}.log'
    shell:
        """
        ./workflow/scripts/mosaic.py {wildcards.field}
        """