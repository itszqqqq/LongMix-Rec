"""用微型 blocked Parquet 检查数据隔离、训练、评估和模型保存。"""
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
import unittest
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
import yaml

import model_zoo
from model_zoo.LONGER_RankMixer import LongerEncoder, MaskedMeanReadout
from unirank.pytorch.layers.interactions.per_token_ffn import PerTokenMoEFFN
from unirank.pytorch.layers.interactions.token_mix import MultiHeadTokenMixing
from unirank.pytorch.models.rank_model import Float32Sigmoid
from unirank.features import FeatureMap
from unirank.preprocess import FeatureProcessor, build_dataset
from unirank.pytorch.dataloaders import RankDataLoader
from unirank.utils import load_config


LABELS = ['is_click', 'is_follow', 'is_like', 'is_comment', 'is_forward', 'long_view']
ROOT = Path(__file__).resolve().parents[1]
CASES = [('LONGER', {}), ('RankMixer', {}), ('TokenMixer', {}),
         ('LONGER_RankMixer', {'rankmixer_num_layers': 0}),
         ('LONGER_RankMixer', {}),
         ('LONGER_RankMixer', {'rankmixer_num_experts': 4, 'rankmixer_num_layers': 2})]


def make_fixture(root):
    """生成两组独立的局部索引，覆盖空历史、有效曝光和含未来事件的轨迹。"""
    dataset = root / 'tiny'
    dataset.mkdir()
    (dataset / 'meta_data.json').write_text(json.dumps({
        'action_vocab': {'exposure': 1, 'is_click|is_like': 2, 'long_view': 3}
    }), encoding='utf8')
    columns = [dict(name=name, active=True, dtype='int', type='meta')
               for name in ('user_index', 'item_index', 'seq_len')]
    columns += [dict(name='user_id', active=True, dtype='int', type='categorical', vocab_size=8),
                dict(name='item_id', active=True, dtype='int', type='categorical', source='item', vocab_size=64),
                dict(name='category', active=True, dtype='int', type='categorical', source='item', vocab_size=8),
                dict(name='action', active=True, dtype='int', type='categorical', source='action', vocab_size=4)]
    options = dict(dataset_id='tiny', data_root=str(root), data_format='parquet', blocked=True,
                   rebuild_dataset=False, feature_cols=columns,
                   label_col=[dict(name=name, dtype='float') for name in LABELS])
    for split in ('train', 'valid', 'test'):
        for folder in ('data', 'user_info', 'item_info'):
            target = dataset / split / folder
            target.mkdir(parents=True)
            options[f'{split}_{folder}'] = str(target)
        for part in range(2):
            filename = f'part-{part:05d}.parquet'
            size = 6 if split == 'train' else 5
            seq_lens = np.array([0, 1, 2, 3, 0, 1])[:size]
            frame = dict(user_index=np.array([0, 0, 0, 0, 1, 1])[:size],
                         item_index=seq_lens + 1, seq_len=seq_lens,
                         user_id=np.array([1, 1, 1, 1, 2, 2])[:size] + part * 2)
            frame.update({name: (np.arange(size) + task) % 2 for task, name in enumerate(LABELS)})
            pd.DataFrame(frame).to_parquet(dataset / split / 'data' / filename, index=False)
            pd.DataFrame(dict(user_index=[0, 1], full_item_seq=[[1, 2, 3, 4]] * 2,
                              full_action_seq=[[1, 2, 3, 2]] * 2)).to_parquet(
                                  dataset / split / 'user_info' / filename, index=False)
            pd.DataFrame(dict(item_index=np.arange(5), item_id=np.array([0, 1, 2, 3, 4]) +
                              np.array([0, 10, 10, 10, 10]) * part,
                              category=[0, 1, 2, 1, 2])).to_parquet(
                                  dataset / split / 'item_info' / filename, index=False)
    encoder = FeatureProcessor(**options)
    build_dataset(encoder, **options)
    feature_map = FeatureMap('tiny', str(dataset))
    feature_map.load(str(dataset / 'feature_map.json'), {'embedding_dim': 4})
    return feature_map, options


def model_options(root):
    """四个模型使用相同的小尺寸和六任务训练配置。"""
    return dict(embedding_dim=4, token_dim=8, max_len=4, num_groups=2, query_size=2,
                num_ns_token=2, num_global_tokens=2, num_rank_tokens=2, num_group_token=2,
                num_heads=2, num_layers=1, longer_num_layers=1, rankmixer_num_layers=1,
                expansion_factor=2, tower_hidden_units=[8], num_tasks=6,
                task='binary_classification', loss=['binary_crossentropy'] * 6,
                dense_optimizer='AdamW', dense_learning_rate=1e-3,
                sparse_optimizer='Adagrad', sparse_learning_rate=0.01,
                model_root=str(root / 'checkpoints'), enable_torch_compile=False,
                enable_bf16=False, verbose=0, metrics=['AUC', 'logloss'],
                accumulation_steps=4, monitor='AUC', eval_steps=None)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """限制 CPU 线程，让微型测试在开发机上快速完成。"""
        torch.set_num_threads(1)

    def setUp(self):
        # Normal inherited permissions also work in restricted Windows sandboxes.
        base = Path(os.environ.get('UNIRANK_TEST_OUTPUT', ROOT / '.test_artifacts')).resolve()
        base.mkdir(parents=True, exist_ok=True)
        self.root = base / uuid4().hex
        self.root.mkdir()
        def cleanup():
            if self.root.resolve().parent != base:
                raise RuntimeError('Refusing cleanup outside test directory')
            shutil.rmtree(self.root)
        self.addCleanup(cleanup)
        self.fm, self.data = make_fixture(self.root)

    def loaders(self, workers=0):
        """走正式的数据入口，验证和测试必须保留不足一个 batch 的尾部。"""
        return RankDataLoader(self.fm, batch_size=2, max_len=4, num_workers=workers,
                              shuffle=False, **self.data).make_iterator()

    def test_data_prefix_labels_and_block_isolation(self):
        train, valid, test = self.loaders()
        batches = list(train)
        self.assertEqual(sum(batch[2].shape[0] for batch in valid), 10)
        self.assertEqual(sum(batch[2].shape[0] for batch in test), 10)
        first, items, mask, tasks = batches[0]
        torch.testing.assert_close(items['item_id'], torch.tensor([[0, 0, 0, 0, 1], [0, 0, 0, 1, 2]]))
        self.assertEqual(mask.sum().item(), 1)
        self.assertEqual(tasks[0].sum().item(), 0)  # exposure 有效，但不是 click。
        self.assertTrue((items['action'][:, -1] == 0).all())
        self.assertEqual(batches[3][1]['item_id'][0, -1].item(), 11)
        # 目标反馈标签变化不能改变输入历史或目标 action。
        path = Path(self.data['train_data']) / 'part-00000.parquet'
        frame = pd.read_parquet(path)
        for label in LABELS:
            frame[label] = 1 - frame[label]
        frame.to_parquet(path, index=False)
        changed = next(iter(self.loaders()[0]))
        for key in items:
            torch.testing.assert_close(items[key], changed[1][key], rtol=0, atol=0)

    def test_worker_loading(self):
        """验证 spawn worker 可序列化，且每个 block 只被消费一次。"""
        train, _, _ = self.loaders(workers=2)
        self.assertEqual(sum(batch[2].shape[0] for batch in train), 12)

    def test_preencoded_large_vocab_needs_no_token_dictionary(self):
        options = dict(self.data)
        options['feature_cols'] = [dict(col) for col in options['feature_cols']]
        options['feature_cols'][4]['vocab_size'] = 32_038_726
        encoder = FeatureProcessor(**options)
        build_dataset(encoder, **options)
        self.assertEqual(encoder.feature_map.features['item_id']['vocab_size'], 32_038_726)
        self.assertEqual(encoder.processor_dict, {})

    def test_models_train_evaluate_reload_and_final_accumulation(self):
        """不足一个累积窗口的梯度也要进入最佳 checkpoint，加载后预测一致。"""
        train, valid, test = self.loaders()
        inputs = next(iter(valid))
        for name, overrides in CASES:
            with self.subTest(model=name, **overrides):
                torch.manual_seed(2026)
                options = dict(model_options(self.root), **overrides)
                model = getattr(model_zoo, name)(self.fm, model_id=name, **options)
                initial = {key: value.clone() for key, value in model.state_dict().items()}
                updates = []
                optimizer_step = model._optimizer_step

                def record_update():
                    optimizer_step()
                    updates.append({key: value.clone() for key, value in model.state_dict().items()})

                model._optimizer_step = record_update
                model.fit(train, epochs=1, validation_data=valid)
                state = torch.load(model.checkpoint, map_location='cpu', weights_only=True)
                self.assertTrue(any(not torch.equal(initial[key], value) for key, value in state.items()))
                self.assertTrue(all(torch.equal(updates[-1][key], value) for key, value in state.items()))
                # 六个 micro-batch、accumulation_steps=4，应执行两次优化。
                self.assertEqual(next(iter(model.dense_optimizer.state.values()))['step'].item(), 2)
                logs = model.evaluate(test)
                self.assertTrue(all(np.isfinite(value) for value in logs.values()))
                self.assertEqual(len(logs), 14)
                model.eval()
                expected = model(inputs)
                restored = getattr(model_zoo, name)(self.fm, model_id=name, **options)
                restored.load_weights(model.checkpoint)
                restored.eval()
                for key, value in restored(inputs).items():
                    torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
                self.assertTrue(all(torch.equal(model.state_dict()[key], value) for key, value in state.items()))

    def test_masked_history_and_causality(self):
        """padding 值不影响预测；历史 query 不读取未来组和候选信息。"""
        encoder = LongerEncoder(8, 4, token_dim=8, max_len=8, num_groups=2,
                                query_size=4, num_global_tokens=2, num_heads=2, num_layers=2).eval()
        history = torch.randn(2, 8, 8)
        context = torch.randn(2, 12)
        mask = torch.tensor([[0] * 8, [0, 0, 1, 1, 1, 1, 1, 1]], dtype=torch.bool)
        encoded, valid = encoder(history, context, mask)
        altered = history.clone()
        altered[~mask] = 1000
        actual, _ = encoder(altered, context, mask)
        torch.testing.assert_close(encoded, actual, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(MaskedMeanReadout()(encoded, valid)).all())
        altered[:, -2:] += 100
        future, _ = encoder(altered, context + 100, mask)
        torch.testing.assert_close(encoded[:, :3], future[:, :3], rtol=0, atol=0)

    def test_cli_full_cycle(self):
        """实际调用训练脚本，覆盖 YAML、BF16 开关和测试后 checkpoint 保留。"""
        config = self.root / 'config'
        config.mkdir()
        (config / 'dataset_config.yaml').write_text(yaml.safe_dump({'tiny': self.data}), encoding='utf8')
        experiments = yaml.safe_load((ROOT / 'config/model_config.yaml').read_text(encoding='utf8'))
        for expid in sorted(experiments.keys() - {'Base'}):
            with self.subTest(expid=expid):
                options = load_config(str(ROOT / 'config'), expid)
                rank_layers = options.get('rankmixer_num_layers', 1)
                options.update(model_options(self.root))
                options.update(dataset_id='tiny', batch_size=2, num_workers=0, epochs=1,
                               shuffle=False, seed=2026, enable_bf16=True, note='中文配置',
                               rankmixer_num_layers=rank_layers)
                (config / 'model_config.yaml').write_text(
                    yaml.safe_dump({expid: options}, allow_unicode=True), encoding='utf8')
                command = [sys.executable, str(ROOT / 'run_expid.py'), '--config', str(config),
                           '--expid', expid, '--gpu', '-1', '--no-enable_bf16']
                run = subprocess.run(command, capture_output=True, text=True, encoding='utf8',
                    env=dict(os.environ, PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1',
                             OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'), timeout=120)
                self.assertEqual(run.returncode, 0, run.stdout[-2000:] + run.stderr[-2000:])
                self.assertIn('BF16 mixed precision: False', run.stdout + run.stderr)
                self.assertIn('******** Test evaluation ********', run.stdout + run.stderr)
                self.assertTrue((self.root / f'checkpoints/tiny/{expid}.model').is_file())

    def test_all_experiment_configs(self):
        """实际实验配置缩小尺寸后，均可构建模型并完成反向传播。"""
        experiments = yaml.safe_load((ROOT / 'config/model_config.yaml').read_text(encoding='utf8'))
        inputs = next(iter(self.loaders()[0]))
        for experiment in experiments.keys() - {'Base'}:
            with self.subTest(experiment=experiment):
                options = load_config(str(ROOT / 'config'), experiment)
                self.assertFalse(options['enable_torch_compile'])
                rank_layers = options.get('rankmixer_num_layers', 1)
                options.update(model_options(self.root))
                options['rankmixer_num_layers'] = rank_layers
                model = getattr(model_zoo, options['model'])(self.fm, **options)
                outputs = model(inputs)
                self.assertTrue(all(f'{label}_pred' in outputs for label in LABELS))
                loss = model.compute_loss(outputs, model.get_labels(inputs))
                self.assertTrue(torch.isfinite(loss))
                loss.backward()

    @unittest.skipUnless(torch.cuda.is_available(), '需要 CUDA GPU')
    def test_cuda_bf16_and_activation_checkpointing(self):
        """在实际 GPU 上检查 BF16 与激活重算的前向、反向和优化器更新。"""
        train, valid, _ = self.loaders()
        for name, overrides in CASES:
            with self.subTest(model=name, **overrides):
                options = dict(model_options(self.root), **overrides)
                options.update(gpu=0, enable_bf16=True,
                               gradient_checkpointing=True)
                model = getattr(model_zoo, name)(self.fm, model_id=name, **options)
                model.fit(train, epochs=1, validation_data=valid)
                self.assertTrue(all(np.isfinite(value) for value in model.evaluate(valid).values()))
                del model
        torch.cuda.empty_cache()

    def test_moe_dual_route_loss_and_gradients(self):
        inputs = next(iter(self.loaders()[0]))
        model = model_zoo.LONGER_RankMixer(self.fm, **dict(model_options(self.root),
            rankmixer_num_experts=4, rankmixer_num_layers=2))
        model.train()
        outputs = model(inputs)
        self.assertEqual(len(outputs), 13)
        labels = model.get_labels(inputs)
        infer_loss = sum(torch.nn.functional.binary_cross_entropy(outputs[f'{k}_pred'], v)
                         for k, v in zip(LABELS, labels))
        train_loss = sum(torch.nn.functional.binary_cross_entropy(outputs[f'{k}_train_pred'], v)
                         for k, v in zip(LABELS, labels))
        loss = model.compute_loss(outputs, labels)
        torch.testing.assert_close(loss, .5 * (infer_loss + train_loss) + outputs['moe_reg_loss'])
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for ffn in model.rankmixer.pffn_layers:
            self.assertGreater(ffn.router_train.weight.grad.abs().sum().item(), 0)
            self.assertGreater(ffn.router_infer.router.weight.grad.abs().sum().item(), 0)
        model.eval()
        self.assertEqual(set(model(inputs)), {f'{k}_pred' for k in LABELS})

    def test_moe_regularizer_only_updates_infer_router(self):
        ffn = PerTokenMoEFFN(2, 8, num_experts=4)
        # Positive logits make the expected L1 gradient unambiguous.
        with torch.no_grad():
            ffn.router_infer.router.weight.zero_()
            ffn.router_infer.router.bias.fill_(1)
        x = torch.randn(3, 2, 8, requires_grad=True)
        _, reg = ffn(x)
        reg.backward()
        self.assertIsNone(x.grad)
        self.assertIsNone(ffn.W1.grad)
        self.assertIsNone(ffn.router_train.weight.grad)
        self.assertGreater(ffn.router_infer.router.bias.grad.abs().sum().item(), 0)
        ffn.eval()
        self.assertEqual(ffn(x)[1].item(), 0)
        self.assertEqual(ffn(x, route='train')[1].item(), 0)
        with self.assertRaises(ValueError):
            ffn(x, route='typo')
        with self.assertRaises(ValueError):
            ffn(x[:, :1])

    def test_bf16_probability_keeps_confident_wrong_gradient(self):
        logits = torch.tensor([6.25], dtype=torch.bfloat16, requires_grad=True)
        probability = Float32Sigmoid()(logits)
        self.assertEqual(probability.dtype, torch.float32)
        self.assertLess(probability.item(), 1)
        loss = torch.nn.functional.binary_cross_entropy(probability, torch.zeros(1))
        loss.backward()
        self.assertGreater(logits.grad.item(), .9)

    def test_token_mixing_keeps_every_coordinate(self):
        for tokens, dim in [(3, 8), (4, 8)]:
            x = torch.arange(tokens * dim, dtype=torch.float32).reshape(1, tokens, dim)
            mixed = MultiHeadTokenMixing(dim, tokens)(x)
            self.assertEqual(mixed.shape, x.shape)
            torch.testing.assert_close(mixed.flatten().sort().values, x.flatten())

    def test_encoder_rejects_invalid_windows(self):
        with self.assertRaises(ValueError):
            LongerEncoder(8, 4, max_len=8, num_groups=2, query_size=5)
        encoder = LongerEncoder(8, 4, token_dim=8, max_len=8, num_groups=2,
                                query_size=2, num_global_tokens=2)
        with self.assertRaises(ValueError):
            encoder(torch.randn(2, 10, 8), torch.randn(2, 12), torch.ones(2, 10))


if __name__ == '__main__':
    unittest.main()
