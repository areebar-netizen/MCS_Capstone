-- DROP TABLE IF EXISTS user_summary;
-- DROP TABLE IF EXISTS user_feedback;
-- DROP TABLE IF EXISTS recommendation;
-- DROP TABLE IF EXISTS model_inference;
-- DROP TABLE IF EXISTS raw_session_data;
-- DROP TABLE IF EXISTS user_profile;

CREATE TABLE user_profile (
    user_id                  VARCHAR(36) PRIMARY KEY,
    first_name               VARCHAR(255),
    last_name                VARCHAR(255),
    email_address            VARCHAR(255),
    country                  VARCHAR(50),
    date_of_birth            DATE,
    gender                   VARCHAR(10),
    occupation               VARCHAR(50),
    education_level          VARCHAR(50),
    primary_focus_goal       VARCHAR(50),
    preferred_stimulus_types VARCHAR(50),
    avoided_stimulus_types   VARCHAR(50),
    sleep_quality            VARCHAR(20)
);

-- csv formatted
CREATE TABLE raw_session_data (
    record_id          INT AUTO_INCREMENT PRIMARY KEY,
    session_id         VARCHAR(36) NOT NULL,
    user_id            VARCHAR(36) NOT NULL,
    session_start_time TIMESTAMP,
    session_end_time   TIMESTAMP,
    timezone           VARCHAR(20),
    session_duration   DECIMAL(6,2),
    session_location   VARCHAR(20),
    phone_present      BOOLEAN,
    energy_level_pre   INT,
    stress_level_pre   INT,
    time_since_waking_hours INT,
    FOREIGN KEY (user_id) REFERENCES user_profile(user_id)
);


CREATE TABLE model_inference (
    inference_id        VARCHAR(36) PRIMARY KEY,
    user_id             VARCHAR(36) NOT NULL,
    session_id          VARCHAR(36) NOT NULL,
    focus_score         DECIMAL(5,2),
    focus_state         VARCHAR(20),
    confidence_score    DECIMAL(5,2),
    model_name          VARCHAR(100),
    model_version       VARCHAR(50),
    focus_drop_detected BOOLEAN,
    concentration_seconds INT,
    neutral_seconds INT,
    relaxed_seconds INT,
    FOREIGN KEY (user_id) REFERENCES user_profile(user_id)
);

CREATE TABLE recommendation (
    recommendation_id       VARCHAR(36) PRIMARY KEY,
    user_id                 VARCHAR(36) NOT NULL,
    session_id              VARCHAR(36) NOT NULL,
    inference_id            VARCHAR(36) NOT NULL,
    recommendation_category VARCHAR(20),
    stimulus_name           VARCHAR(20),
    trigger_reason          VARCHAR(20),
    action_started_at       TIMESTAMP,
    action_ended_at         TIMESTAMP,
    action_duration_minutes DECIMAL(4,2),
    message                 VARCHAR(255),
    FOREIGN KEY (user_id)      REFERENCES user_profile(user_id),
    FOREIGN KEY (inference_id) REFERENCES model_inference(inference_id)
);


CREATE TABLE user_feedback (
    feedback_id              VARCHAR(36) PRIMARY KEY,
    user_id                  VARCHAR(36) NOT NULL,
    session_id               VARCHAR(36) NOT NULL,
    inference_id             VARCHAR(36) NOT NULL,
    recommendation_id        VARCHAR(36),
    feedback_type            VARCHAR(50),
    `trigger`                 VARCHAR(100),
    helpfulness_rating       INT,
    ease_of_use_rating       INT,
    recommendation_relevance INT,
    overall_rating           INT,
    sentiment                VARCHAR(20),
    FOREIGN KEY (user_id)           REFERENCES user_profile(user_id),
    FOREIGN KEY (inference_id)      REFERENCES model_inference(inference_id),
    FOREIGN KEY (recommendation_id) REFERENCES recommendation(recommendation_id)
);

CREATE TABLE user_summary (
    user_id                     VARCHAR(36) PRIMARY KEY,
    total_sessions              INT,
    total_focus_time_minutes    DECIMAL(6,2),
    average_focus_score         DECIMAL(6,2),
    best_focus_score            DECIMAL(6,2),
    optimal_focus_time_of_day   VARCHAR(20),
    first_session_date          DATE,
    last_session_date           DATE,
    stimulus_used               VARCHAR(50),
    recommendation_feedback     VARCHAR(50),
    most_effective_stimulus     VARCHAR(50),
    least_effective_stimulus    VARCHAR(50),
    average_feedback_rating     INT,
    positive_feedback_count     INT,
    negative_feedback_count     INT,
    overall_sentiment_score     DECIMAL(5,2),
    total_recommendations_rated INT,
    FOREIGN KEY (user_id) REFERENCES user_profile(user_id)
);
