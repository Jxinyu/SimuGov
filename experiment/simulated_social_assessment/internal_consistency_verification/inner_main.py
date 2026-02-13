import time

from eva_compare import eva_compare
from eva_robustness import eva_robustness


if __name__ == '__main__':
    # low_beta_file = r'experiment\simulated_social_assessment\internal_consistency_verification\data\eva_compare\test-2\low\惩罚0_99_教育低_ai_threshold_0_01'
    # high_beta_file = r'experiment\simulated_social_assessment\internal_consistency_verification\data\eva_compare\test-2\high\惩罚0_99_教育低_ai_threshold_0_01'
    output_dir = fr'experiment\Simulated social assessment\Internal consistency verification\output\{str(time.time()).split(".")[0]}'
    # day_time = 15
    # eva_compare(low_beta_file, high_beta_file, output_dir, day_time)
    #
    robustness_data = r'experiment\Simulated social assessment\Internal consistency verification\data\eva_robustness'
    eva_robustness(robustness_data, output_dir)

