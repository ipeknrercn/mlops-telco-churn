import pytest
import pandas as pd


def create_feature_crosses_standalone(df):
    """Standalone version for testing - no external dependencies"""
    df["cross_contract_payment"] = (
        df["contract_type"].astype(str) + "__x__" + df["payment_method"].astype(str)
    )
    df["cross_service_contract"] = (
        df["service_combo_id"].astype(str) + "__x__" + df["contract_type"].astype(str)
    )
    df["cross_geo_contract"] = (
        df["geo_code"].astype(str) + "__x__" + df["contract_type"].astype(str)
    )
    return df


class TestFeatureCrosses:
    def test_feature_cross_creation(self):
        """Test that feature crosses are created correctly"""
        df = pd.DataFrame(
            {
                "contract_type": ["Month-to-month"],
                "payment_method": ["Electronic check"],
                "service_combo_id": ["ServiceA"],
                "geo_code": ["G01"],
            }
        )
        result = create_feature_crosses_standalone(df)

        assert "cross_contract_payment" in result.columns
        assert (
            result["cross_contract_payment"][0]
            == "Month-to-month__x__Electronic check"
        )

    def test_feature_cross_count(self):
        """Test that exactly 3 feature crosses are created"""
        df = pd.DataFrame(
            {
                "contract_type": ["One year", "Two year"],
                "payment_method": ["Bank transfer", "Credit card"],
                "service_combo_id": ["ServiceB", "ServiceC"],
                "geo_code": ["G02", "G03"],
            }
        )
        result = create_feature_crosses_standalone(df)

        cross_columns = [col for col in result.columns if col.startswith("cross_")]
        assert len(cross_columns) == 3

    def test_feature_cross_deterministic(self):
        """Test that feature crosses are deterministic"""
        df1 = pd.DataFrame(
            {
                "contract_type": ["Month-to-month"],
                "payment_method": ["Electronic check"],
                "service_combo_id": ["ServiceA"],
                "geo_code": ["G01"],
            }
        )
        df2 = df1.copy()

        result1 = create_feature_crosses_standalone(df1)
        result2 = create_feature_crosses_standalone(df2)

        assert (
            result1["cross_contract_payment"][0]
            == result2["cross_contract_payment"][0]
        )


class TestDataValidation:
    def test_missing_values_handling(self):
        """Test that None values are handled correctly"""
        df = pd.DataFrame(
            {
                "contract_type": [None, "One year"],
                "payment_method": ["Electronic check", None],
                "service_combo_id": ["ServiceA", "ServiceB"],
                "geo_code": ["G01", "G02"],
            }
        )
        result = create_feature_crosses_standalone(df)

        # None should be converted to string 'None'
        assert "None" in str(result["cross_contract_payment"][0])



