use time::format_description;
use tracing_appender::{non_blocking, rolling};
use tracing_error::ErrorLayer;
use tracing_subscriber::{
    fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer, Registry,
};

pub fn init_logger(log_level: String, log_file: String) -> anyhow::Result<non_blocking::WorkerGuard> {
    let timer = fmt::time::LocalTime::new(format_description::well_known::Rfc3339);
    let filter = EnvFilter::from_default_env()
        .add_directive(format!("md5rs={}", log_level.to_lowercase()).parse()?)
        .add_directive("nom-exif=off".parse()?); // 环境变量过滤器

    let formatting_layer = fmt::layer()
        .pretty()
        .with_timer(timer.clone())
        .with_writer(std::io::stderr);

    let file_appender = rolling::daily("logs/", log_file.as_str()); // 每天一个日志文件
    let (non_blocking_appender, guard) = non_blocking(file_appender); // 输出非阻塞
    let file_layer = fmt::layer()
        .with_ansi(false)
        .with_timer(timer)
        .with_writer(non_blocking_appender) // 文件输出日志等级
        .boxed();

    Registry::default()
        .with(filter)
        .with(ErrorLayer::default())
        .with(formatting_layer)
        .with(file_layer)
        .init();

    Ok(guard)
}
