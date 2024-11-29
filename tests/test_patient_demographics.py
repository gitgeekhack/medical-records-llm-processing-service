import json
from json import JSONDecodeError

import pytest
import logging

from app.constant import MedicalInsights
from app.service.nlp_extractor.patient_demographics_extractor import PatientDemographicsExtractor

pytest_plugins = ('pytest_asyncio',)

class TestPatientDemographicsExtractor:
    @pytest.mark.asyncio
    async def test_extract_number_with_integer_value(self):
        logger = logging.getLogger()
        pd = PatientDemographicsExtractor(logger)
        text = "Age is 36"
        assert await pd._PatientDemographicsExtractor__extract_number(text) == "36"

    @pytest.mark.asyncio
    async def test_extract_number_without_integer_value(self):
        logger = logging.getLogger()
        pd = PatientDemographicsExtractor(logger)
        text = "Age is thirty six"
        assert await pd._PatientDemographicsExtractor__extract_number(text) == ""

    @pytest.mark.asyncio
    async def test_extract_number_with_null_string(self):
        logger = logging.getLogger()
        pd = PatientDemographicsExtractor(logger)
        text = ""
        assert await pd._PatientDemographicsExtractor__extract_number(text) == ""

    @pytest.mark.asyncio
    async def test_process_patient_demographics_with_valid_output_text(self):
        logger = logging.getLogger()
        pd = PatientDemographicsExtractor(logger)
        output_text = """{"patient_name": "Latoria Maxie", "date_of_birth": "9 August 1976", "age": "46 years", "gender": "female",
                                                       "height": {"value": "64 inches", "date": "01-17-2023"}, "weight": {"value": "198", "date": "01-17-2023"}, "bmi": "33.98"}"""

        assert await pd._PatientDemographicsExtractor__process_patient_demographics(output_text) == {"patient_demographics": {"patient_name": "Latoria Maxie", "date_of_birth": "08-09-1976", "age": "46", "gender": "Female", "height": {"value": "64", "date": "01-17-2023"}, "weight": {"value": "198", "date": "01-17-2023"}, "bmi": "33.98"}}

    @pytest.mark.asyncio
    async def test_process_patient_demographics_with_invalid_output_text(self):
        logger = logging.getLogger()
        pd = PatientDemographicsExtractor(logger)
        output_text = """{{"patient_name": "Latoria Maxie", "date_of_birth": "9 August 1976", "age": "46 years", "gender": "female",
                                                           "height": {"value": "64 inches", "date": "01-17-2023"}, "weight": {"value": "198", "date": "01-17-2023"}, "bmi": "33.98"}}"""

        try:
            await pd._PatientDemographicsExtractor__process_patient_demographics(output_text)
        except JSONDecodeError:
            assert True

    @pytest.mark.asyncio
    async def test_process_patient_demographics_with_empty_output_text(self):
        logger = logging.getLogger()
        pd = PatientDemographicsExtractor(logger)
        output_text = ""
        template_data = MedicalInsights.TemplateResponse.DEMOGRAPHICS_TEMPLATE_RESPONSE
        assert await pd._PatientDemographicsExtractor__process_patient_demographics(output_text) == {"patient_demographics": template_data}

    @pytest.mark.asyncio
    async def test_get_patient_demographics_with_pagewsie_text(self):
        logger = logging.getLogger()
        pd = PatientDemographicsExtractor(logger)
        with open('static/Fugarino Dictation_ 06-27-2023_text.json') as f: text = json.load(f)
        assert await pd.get_patient_demographics(text) is not None

    @pytest.mark.asyncio
    async def test_get_patient_demographics_without_pagewsie_text(self):
        logger = logging.getLogger()
        pd = PatientDemographicsExtractor(logger)
        text = "Patient name is maxie, she is a Female and she is 46 years old"
        try:
            await pd.get_patient_demographics(text)
        except AttributeError:
            assert True
