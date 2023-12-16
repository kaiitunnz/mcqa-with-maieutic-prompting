palm_api_key = "api_key"
model = "models/text-bison-001"

belief_candidate_count = 8  # Must be in the interval [1, 8].
belief_temp = 0.3
belief_thresh = 0.2
first_level_candidate = 3
first_level_temp = 0.5
second_level_candidate = 1
second_level_temp = 0.5
max_depth = 2

save_period = 1  # Period to save the generated trees to disk.
