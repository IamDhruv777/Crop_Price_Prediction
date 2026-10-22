import pandas as pd
from typing import Tuple, Dict, Any
from enum import Enum

class ValidationStatus(Enum):
    VALID = 'valid'
    FLAGGED = 'flagged'
    INVALID = 'invalid'

class QualityFlag(Enum):
    PRICE_NEGATIVE = 'price_negative'
    PRICE_INCONSISTENT = 'price_inconsistent'
    PRICE_ZERO = 'price_zero'
    DATE_INVALID = 'date_invalid'
    OUTLIER_PRICE = 'outlier_price'
    MISSING_REQUIRED = 'missing_required'

class DataValidator:
    def validate_price_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        required_columns = ['state', 'district', 'market', 'commodity', 'price_date', 'min_price', 'max_price', 'modal_price']
        
        report = {
            'total_rows': len(df),
            'missing_columns': [],
            'flags': {flag.value: 0 for flag in QualityFlag}
        }
        
        validated_df = df.copy()
        validated_df['validation_status'] = ValidationStatus.VALID.value
        validated_df['quality_flags'] = [[] for _ in range(len(df))]
        
        for col in required_columns:
            if col not in validated_df.columns:
                report['missing_columns'].append(col)
        
        if report['missing_columns']:
            return validated_df, report
            
        # Check negative prices
        neg_mask = (validated_df['min_price'] < 0) | (validated_df['max_price'] < 0) | (validated_df['modal_price'] < 0)
        
        # Check min_price <= max_price
        inconsistent_mask1 = validated_df['min_price'] > validated_df['max_price']
        
        # Flag cases where modal_price < min_price or modal_price > max_price
        inconsistent_mask2 = (validated_df['modal_price'] < validated_df['min_price']) | (validated_df['modal_price'] > validated_df['max_price'])
        
        # Check zeros
        zero_mask = (validated_df['min_price'] == 0) | (validated_df['max_price'] == 0) | (validated_df['modal_price'] == 0)

        def add_flag(mask, flag):
            if mask.any():
                indices = mask[mask].index
                for idx in indices:
                    flags_list = validated_df.at[idx, 'quality_flags']
                    if isinstance(flags_list, list):
                        flags_list.append(flag.value)
                    validated_df.at[idx, 'validation_status'] = ValidationStatus.FLAGGED.value
                report['flags'][flag.value] = int(mask.sum())

        add_flag(neg_mask, QualityFlag.PRICE_NEGATIVE)
        add_flag(inconsistent_mask1 | inconsistent_mask2, QualityFlag.PRICE_INCONSISTENT)
        add_flag(zero_mask, QualityFlag.PRICE_ZERO)
        
        # Date formats
        date_mask = pd.to_datetime(validated_df['price_date'], errors='coerce').isna()
        add_flag(date_mask, QualityFlag.DATE_INVALID)
        
        # Missing required
        missing_mask = validated_df[required_columns].isna().any(axis=1)
        add_flag(missing_mask, QualityFlag.MISSING_REQUIRED)

        return validated_df, report

    def validate_weather_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        report = {
            'total_rows': len(df),
            'flags': {}
        }
        validated_df = df.copy()
        return validated_df, report

    def validate_arrival_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        report = {
            'total_rows': len(df),
            'flags': {}
        }
        validated_df = df.copy()
        return validated_df, report

    def validate_coordinates(self, lat: float, lon: float) -> bool:
        if pd.isna(lat) or pd.isna(lon):
            return False
        if not (8.0 <= lat <= 37.0):
            return False
        if not (68.0 <= lon <= 97.0):
            return False
        return True

    def generate_quality_report(self, df: pd.DataFrame, validation_results: dict) -> dict:
        total = validation_results.get('total_rows', len(df))
        flags = validation_results.get('flags', {})
        total_issues = sum(flags.values())
        
        score = 100
        if total > 0:
            score = max(0, 100 - (total_issues / total * 100))
            
        return {
            'quality_score': score,
            'total_records_checked': total,
            'total_issues_found': total_issues,
            'issue_breakdown': flags
        }
