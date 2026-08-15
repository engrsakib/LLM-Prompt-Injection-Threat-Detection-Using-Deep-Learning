# Reproducibility Appendix

Generated: `2026-08-15T08:48:23.300009+00:00`

## Environment

- Python: `3.13.7`
- Platform: `Windows-11-10.0.22631-SP0`
- Git commit: `97bc17c0b2d207451c94a572f70940d211511817`

## Exact Commands

```bash
pip install -r requirements.txt
```

```bash
python -m src.data.prepare --config configs/data.yaml --phase 1
```

```bash
python -m src.data.enrich --config configs/data.yaml
```

```bash
python -m src.data.finalize --config configs/data.yaml
```

```bash
dvc repro prepare
```

```bash
dvc repro enrich
```

```bash
dvc repro finalize
```

## Artifact Hashes (SHA-256)

- `data/processed/metadata.json`: `c99ce8cd6cb0c69422c24bbe1c73100d14843fcc4b8fa07418aff1b20f4cf68b`
- `data/processed/reports/ambiguity_subsets.json`: `01666eb7656b7d4612bc698a6a48aae4607d3ea69a7bff443dfc8d7d7221083f`
- `data/processed/reports/class_distribution.json`: `c6313f6e481b95cfeff0857186641a2e63564a23bddedb16bd91dfa9d5b90399`
- `data/processed/reports/dedup_report.json`: `13d7dd585fb45176a3de54e6599e24586d6f9e8d6b3b45ad00e07825bee2d837`
- `data/processed/reports/leakage_report.json`: `1fe0ba9b5ecbdd50af23bb44d3abd15a2c827f019b96d57417592aa77bcbcdc4`
- `data/processed/reports/severity_buckets.json`: `da533ff4a93aeb0404b79f990e19e2f0b10e3ce7443ca077be3a9c9e9b246659`
- `data/processed/reports/technique_subsets.json`: `7707bc350582f4f7c0800121ceec2663284a8e1e22e0f6403e2787cd5f5ee2a2`
- `data/processed/reports/validation_report.json`: `eb59ca97f741a820b543c98a3d7bcf9c648e7f3df02ffd2e5d40f19764ffac9c`
- `data/processed/subsets/ambiguity_test.parquet`: `3a6e541aa3c2a3e0af6d6b188d165722bcb672318f1947943dcae4cf51b056dc`
- `data/processed/subsets/ambiguity_train.parquet`: `2f72ccb8aeac58437edd3b31f6483d86bafa6a8bc3d8dc18296c7bfca7af5e4e`
- `data/processed/subsets/ambiguity_validation.parquet`: `6fa862e6282033bf4d75d0cbc59fb96bcac7161b552fad7766aa812935e5525b`
- `data/processed/subsets/technique_encoding_test.parquet`: `bcca4cbfcc48d9600882a9f7f95624323f995f83a571a3828c65d0531ed9429b`
- `data/processed/subsets/technique_encoding_train.parquet`: `4f8b49225492e3489af5c3b73a514c082c72ee9b02d72eaa7284b25b25f95d02`
- `data/processed/subsets/technique_encoding_validation.parquet`: `e0e08030fc550170129161ade30a434d85b40d1ba66837ab5e122d09af28ed94`
- `data/processed/subsets/technique_role_play_test.parquet`: `2d19441392c56d16fd558b5dd250e9ef1b40349a8a7701fb8b09446968d6e15c`
- `data/processed/subsets/technique_role_play_train.parquet`: `c9f0e8883650e604ad2732e4471a63618f7da2a2402bf0c71e26da5e587e3867`
- `data/processed/subsets/technique_role_play_validation.parquet`: `1ec974d30f0b3abc54aaa53cf491012b6a011cbb0aecc12092ebc3a49cb1961a`
- `data/processed/subsets/technique_tool_abuse_test.parquet`: `12d287dc18a3a269b55564acbf286aad16e60d022abbc34b2c125175a0b4f629`
- `data/processed/subsets/technique_tool_abuse_train.parquet`: `ec5664ebc1ac106c2946f176b9877e3587a1ea88ef31d25493eea43c356d3074`
- `data/processed/subsets/technique_tool_abuse_validation.parquet`: `6ae0fbd5420e045b0aad5620cc631f09e9afda3e41a196d35920787f8d57fed4`
- `data/processed/test.parquet`: `801589445d1b52b79611efd09426eb6d426e15e01f943ce6b3c41f8bb5e0ca11`
- `data/processed/train.parquet`: `c550069f76e7e6ddc31d36df5d1d3e3bff1bbf9bd840cdd260675167e0f509d6`
- `data/processed/train_adversarial.parquet`: `cec058784b1ed0fef7444267baa5b60148a0302595cab907e7dfa34110367c97`
- `data/processed/train_augmented.parquet`: `b519ff23a6aa549c32685ab1812f4fa5961b65c81e6bd62a74d77b45da498862`
- `data/processed/train_balanced.parquet`: `2b7ae0aa1bc4961fcc695e095106242a7010acaba018c11a1140389d72f01a11`
- `data/processed/validation.parquet`: `be75f716b9e67f892deac0c93c1fcfd01f58ad07e3c7511f93ac24f72de3f5a4`