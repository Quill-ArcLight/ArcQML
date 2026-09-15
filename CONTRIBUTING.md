# 贡献

提交前阅读 LICENSE、LICENSE-RUNTIME 和 THIRD_PARTY_NOTICES.md。
仅提交你有权贡献的源码；不得提交私有 Runtime 实现、凭据或无授权第三方材料。

提交贡献表示你确认有权按本项目 LICENSE 授权所贡献的原创内容，保留你自己的
版权；第三方内容须标明来源、版本和原许可。不要声称本声明是标准 DCO，也不要
默认将贡献者权利转让给维护者。另行商业再许可仍需获得相关权利人的授权。

说明变更目的、影响及验证结果。Rust 使用匹配 Runtime 的工具链，运行
`cargo test --workspace --release --locked`；Python wheel 安装后运行
`python -m unittest discover -s python/tests`。安全问题按 SECURITY.md 私下报告。
