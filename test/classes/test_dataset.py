import pytest
from classes import CipherPlainData, Config
import os
import zipfile
from pathlib import Path
import json


@pytest.fixture
def real_data_config(tmp_path):
	"""
	Reads static JSON files from test/test_data, zips them into a temporary
	directory, and returns a Config object pointed at that directory.
	"""
	test_data_dir = Path(__file__).parent.parent / "ciphers"
	zip_path = tmp_path / "ciphers.zip"

	with zipfile.ZipFile(zip_path, "w") as z:
		for json_file in test_data_dir.glob("*.json"):
			z.write(json_file, arcname=json_file.name)

	return Config(data_dir=tmp_path)


@pytest.fixture
def cipher():
	"""Fixture for a cipher."""
	return {
		"plaintext": "test",
		"length": 4,
		"num_symbols": 4,
		"difficulty": 1,
		"key": {
			"e": [2],
			"s": [3],
			"t": [1],
		},
		"source_id": "12345",
		"source_name": "test",
		"ciphertext": "1 2 3 1",
	}


def write_cipher(data_dir, cipher):
	zip1_path = data_dir / "batch_01.zip"
	with zipfile.ZipFile(zip1_path, "w") as z:
		z.writestr("cipher_a.json", json.dumps(cipher))


@pytest.fixture
def cipher_illegal_char(tmp_path, cipher):
	"""Fixture for a cipher with an illegal character."""
	data_dir = tmp_path / "data"
	data_dir.mkdir()

	cipher["ciphertext"] = "1 2 3 1 4"
	cipher["key"]["æ"] = [4]
	cipher["plaintext"] = "testæ"

	write_cipher(data_dir, cipher)

	return Config(data_dir=data_dir)


@pytest.fixture
def cipher_invalid_json(tmp_path, cipher):
	"""Fixture for a cipher with invalid JSON (missing key)."""
	data_dir = tmp_path / "data"
	data_dir.mkdir()

	del cipher["plaintext"]

	write_cipher(data_dir, cipher)

	return Config(data_dir=data_dir)


class TestCipherPlainDataInit:
	def test_init(self):
		data = CipherPlainData(Config())
		assert data.file_refs == []
		assert data.handles == {}
		assert data.sep_token == data.config.unique_homophones + 1
		assert data.space_token == data.sep_token + 1
		assert data.char_offset == data.space_token + 1

	def test_init_with_file_refs(self, tmp_path):
		"""Test that the CipherPlainData class can be initialized with file refs."""
		data_dir = tmp_path / "data"
		data_dir.mkdir()

		zip1_path = data_dir / "batch_01.zip"
		with zipfile.ZipFile(zip1_path, "w") as z:
			z.writestr("cipher_a.json", '{"dummy": "data"}')
			z.writestr("cipher_b.json", '{"dummy": "data"}')
			z.writestr("ignore_me.txt", "This should not be indexed")

		zip2_path = data_dir / "batch_02.zip"
		with zipfile.ZipFile(zip2_path, "w") as z:
			z.writestr("nested_folder/cipher_c.json", '{"dummy": "data"}')

		(data_dir / "not_a_zip.tar.gz").write_text("fake archive")

		dataset = CipherPlainData(Config(data_dir=data_dir))

		assert len(dataset.file_refs) == 3
		assert len(dataset) == 3  # Test length simultaneously

		actual_refs = {(os.path.basename(zp), fname) for zp, fname in dataset.file_refs}

		expected_refs = {
			("batch_01.zip", "cipher_a.json"),
			("batch_01.zip", "cipher_b.json"),
			("batch_02.zip", "nested_folder/cipher_c.json"),
		}

		assert actual_refs == expected_refs

	def test_length_with_empty_zip(self, tmp_path):
		"""Test that the length of the CipherPlainData class is correct when a zip is empty."""
		data_dir = tmp_path / "data"
		data_dir.mkdir()

		zip1_path = data_dir / "batch_01.zip"
		with zipfile.ZipFile(zip1_path, "w") as z:
			z.writestr("not_a_cipher.txt", "Doesn't matter")

		dataset = CipherPlainData(Config(data_dir=data_dir))

		assert len(dataset) == 0


class TestCipherPlainDataGetItem:
	def test_getitem(self, real_data_config):
		"""Test that the CipherPlainData class can be indexed."""
		dataset = CipherPlainData(real_data_config)

		item = dataset[0]

		assert item["input_ids"].shape == (Config().max_context,)
		assert item["labels"].shape == (Config().max_context,)

	def test_getitem_unexpected_char(self, cipher_illegal_char):
		"""Test that the CipherPlainData class raises an error when an
		unexpected character is encountered."""
		dataset = CipherPlainData(cipher_illegal_char)

		with pytest.raises(ValueError) as e:
			dataset[0]
		assert "Unexpected char" in str(e.value)

	def test_getitem_invalid_json(self, cipher_invalid_json):
		"""Test that the CipherPlainData class raises an error when an
		invalid JSON file is encountered."""
		dataset = CipherPlainData(cipher_invalid_json)

		with pytest.raises(ValueError) as e:
			dataset[0]
		assert "Item is missing key: plaintext" in str(e.value)


class TestCipherPlainDataSpacesInit:
	def test_init_keys_no_spaces(self):
		config = Config(use_spaces=False)
		dataset = CipherPlainData(config)
		assert dataset.text_key == "plaintext"
		assert dataset.cipher_key == "ciphertext"

	def test_init_keys_with_spaces(self):
		config = Config(use_spaces=True)
		dataset = CipherPlainData(config)
		assert dataset.text_key == "plaintext_with_boundaries"
		assert dataset.cipher_key == "ciphertext_with_boundaries"

	def test_token_offsets_consistency(self):
		config = Config(unique_homophones=100)
		dataset = CipherPlainData(config)

		# Test if [CIPHER] + [SEP] + [SPACE] + [CHAR OFFSET]
		assert dataset.sep_token == 101
		assert dataset.space_token == dataset.sep_token + 1
		assert dataset.char_offset == dataset.space_token + 1


class TestCipherPlainDataMapping:
	@pytest.fixture
	def spaced_cipher_item(self):
		return {
            "plaintext": "ab",
            "plaintext_with_boundaries": "a_b",
            "ciphertext": "1 2",
            "ciphertext_with_boundaries": "1 _ 2",
            "length": 2, "num_symbols": 2, "difficulty": 1,
            "key": {"a": [1], "b": [2]},
            "source_id": "1", "source_name": "test"
        }

	def test_getitem_mapping_with_spaces(self, tmp_path, spaced_cipher_item):
		data_dir = tmp_path / "data"
		data_dir.mkdir()
		write_cipher(data_dir, spaced_cipher_item)

		config = Config(data_dir=data_dir, use_spaces=True, unique_homophones=10)
		dataset = CipherPlainData(config)

		item = dataset[0]
		ids = item["input_ids"].tolist()

        # [Cipher: 1, Space, 2] + [SEP] + [Plain: 'a', Space, 'b']
        # IDs: [1, 12, 2, 11, 13, 12, 14]

		assert ids[0] == 1                           # Cipher '1'
		assert ids[1] == dataset.space_token          # Cipher '_'
		assert ids[2] == 2                           # Cipher '2'
		assert ids[3] == dataset.sep_token            # [SEP]
		assert ids[4] == dataset.char_offset          # 'a'
		assert ids[5] == dataset.space_token          # Plain '_'
		assert ids[6] == dataset.char_offset + 1      # 'b'

	def test_getitem_mapping_no_spaces(self, tmp_path, spaced_cipher_item):
		data_dir = tmp_path / "data"
		data_dir.mkdir()
		write_cipher(data_dir, spaced_cipher_item)

		config = Config(data_dir=data_dir, use_spaces=False, unique_homophones=10)
		dataset = CipherPlainData(config)

		item = dataset[0]
		ids = item["input_ids"].tolist()

		# Should ignore the underscores in the JSON
		assert dataset.space_token not in ids[:10] # Space shouldn't be in the active sequence
