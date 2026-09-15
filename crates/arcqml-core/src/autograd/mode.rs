use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

/// 当前线程内处于 no_grad 作用域的标识。
#[derive(Default)]
struct GradModeState {
    next_token: u64,
    active_tokens: Vec<u64>,
}

thread_local! {
    /// 每个线程独立保存梯度记录状态，避免并发任务相互影响。
    static GRAD_MODE_STATE: RefCell<GradModeState> = RefCell::new(GradModeState::default());
}

/// 返回当前线程是否会为可微操作记录自动微分图。
pub fn is_grad_enabled() -> bool {
    GRAD_MODE_STATE.with(|state| state.borrow().active_tokens.is_empty())
}

/// 临时关闭当前线程的自动微分图记录。
pub fn no_grad() -> NoGradGuard {
    let token = GRAD_MODE_STATE.with(|state| {
        let mut state = state.borrow_mut();
        let token = state.next_token;
        state.next_token = state.next_token.wrapping_add(1);
        state.active_tokens.push(token);
        token
    });

    NoGradGuard {
        token,
        // 防止 guard 被移动到创建它的线程之外
        _thread_bound: PhantomData,
    }
}

/// no_grad() 返回的作用域 guard。
pub struct NoGradGuard {
    token: u64,
    _thread_bound: PhantomData<Rc<()>>,
}

impl Drop for NoGradGuard {
    /// 离开作用域时仅移除自身标识，因此嵌套或非 LIFO 释放不会错误恢复梯度记录。
    fn drop(&mut self) {
        GRAD_MODE_STATE.with(|state| {
            let mut state = state.borrow_mut();
            if let Some(index) = state
                .active_tokens
                .iter()
                .position(|&active_token| active_token == self.token)
            {
                state.active_tokens.swap_remove(index);
            }
        });
    }
}
