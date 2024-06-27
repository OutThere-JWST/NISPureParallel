# Stage 1 Rules
for field in FIELDS:
    rule:
        name: f"stage1_{field}"
        input:
            [f'UNCAL/{f}' for f in uncal[field]]
        output:
            [f'RATE/{f.replace('uncal','rate')}' for f in uncal[field]]
        conda:'envs/jwst.yaml'
        log: f'logs/Stage1_{field}.log'
        shell: 
            f"""
            ./workflow/scripts/runStage1.py {field} --ncpu $SLURM_CPUS_PER_TASK
            """