# Task 1 Implementation Checklist

## Dependency/Config Test Checklist

- [x] raw PyTorch only (no Lightning dependency in final requirements)
- [x] minFM as upstream reference
- [x] configurable replacement span (start_layer, end_layer)
- [x] no lightning dependency unless truly needed

## Verification Steps

### Config verification
```bash
cd /home/hungphongtrn/Workspace/midflowlm/midflowlm && python3 -c "import yaml; cfg=yaml.safe_load(open('configs/v0_onemotif.yaml')); print(cfg['replacement_model']['family']); print(cfg['replacement_model']['start_layer'], cfg['replacement_model']['end_layer']); print('train_loop' in cfg)"
```
Expected:
- line 1: `minfm-hidden-refiner`
- line 2: `8 11`
- line 3: `True`

### Requirements verification
```bash
cd /home/hungphongtrn/Workspace/midflowlm/midflowlm && python3 -c "from pathlib import Path; txt=Path('requirements.txt').read_text(); print('lightning' in txt.lower())"
```
Expected: `False`
