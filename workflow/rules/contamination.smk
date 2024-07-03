# Contamination Rule
rule contam:
    input:
        'FIELDS/{field}/logs/mos.out'
    output:
        'FIELDS/{field}/logs/contam.out'
    conda:'../envs/grizli.yaml'
    log:'logs/Contam_{field}.log'
    shell:
        """
        ./workflow/scripts/contamination.py {wildcards.field} --ncpu $SLURM_CPUS_PER_TASK
        """